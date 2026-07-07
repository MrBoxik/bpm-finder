from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly, stft


DEFAULT_MAX_ANALYZE_SECONDS = 35.0
DEFAULT_TARGET_SAMPLE_RATE = 8_000
DEFAULT_DEEP_CONFIDENCE_THRESHOLD = 0.90
DEFAULT_DEEP_WINDOW_SECONDS = 35.0
DEFAULT_DEEP_MAX_WINDOWS = 12


@dataclass(frozen=True)
class BpmResult:
    bpm: float
    confidence: float
    duration_seconds: float
    analyzed_seconds: float


@dataclass(frozen=True)
class _TempoCandidate:
    bpm: float
    score: float
    source: str


class BpmDetectionError(Exception):
    """Raised when the audio cannot produce a useful tempo estimate."""


def estimate_bpm(
    file_path: str | Path,
    *,
    min_bpm: float = 60.0,
    max_bpm: float = 200.0,
    target_sample_rate: int = DEFAULT_TARGET_SAMPLE_RATE,
    max_analyze_seconds: float = DEFAULT_MAX_ANALYZE_SECONDS,
    deep_confidence_threshold: str | float = DEFAULT_DEEP_CONFIDENCE_THRESHOLD,
) -> BpmResult:
    path = Path(file_path)
    if not path.exists():
        raise BpmDetectionError(f"File does not exist: {path}")
    if min_bpm <= 0 or max_bpm <= min_bpm:
        raise ValueError("Expected 0 < min_bpm < max_bpm.")
    deep_threshold = parse_confidence_threshold(deep_confidence_threshold)

    samples, source_sample_rate, duration, analyzed_seconds = _read_audio_segment(
        path,
        max_analyze_seconds=max_analyze_seconds,
    )
    mono = _prepare_mono_audio(
        samples,
        source_sample_rate=source_sample_rate,
        target_sample_rate=target_sample_rate,
    )

    bpm, confidence = _estimate_tempo(
        mono,
        sample_rate=target_sample_rate,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
    )
    primary_bpm = bpm
    primary_confidence = confidence
    if confidence <= deep_threshold and duration > 45.0:
        ensemble = _estimate_tempo_ensemble(
            path,
            duration_seconds=duration,
            segment_seconds=max_analyze_seconds,
            target_sample_rate=target_sample_rate,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
        )
        if ensemble is not None and ensemble[1] >= confidence:
            ensemble_bpm, ensemble_confidence = ensemble
            if _should_keep_primary_tempo(
                primary_bpm,
                primary_confidence,
                ensemble_bpm,
            ):
                bpm = primary_bpm
                confidence = max(primary_confidence, ensemble_confidence)
            else:
                bpm, confidence = ensemble
    return BpmResult(
        bpm=round(float(bpm), 1),
        confidence=_conservative_confidence(float(confidence)),
        duration_seconds=round(float(duration), 2),
        analyzed_seconds=round(float(analyzed_seconds), 2),
    )


def parse_confidence_threshold(value: str | int | float) -> float:
    if isinstance(value, str):
        text = value.strip().replace(" ", "")
        has_percent_sign = text.endswith("%")
        if text.endswith("%"):
            text = text[:-1]
        if not text:
            raise ValueError("Deep Check Below needs a confidence value.")
        try:
            number = float(text)
        except ValueError as exc:
            raise ValueError("Deep Check Below must be a number like 99, 99%, or 0.99.") from exc
    else:
        number = float(value)

    if not math.isfinite(number):
        raise ValueError("Deep Check Below must be a real number.")
    if isinstance(value, str) and has_percent_sign:
        number /= 100.0
    elif number > 1.0:
        number /= 100.0
    if not 0.0 <= number <= 1.0:
        raise ValueError("Deep Check Below must be between 0% and 100%.")
    return number


def format_confidence_threshold(value: str | int | float) -> str:
    percent = parse_confidence_threshold(value) * 100.0
    formatted = f"{percent:.2f}".rstrip("0").rstrip(".")
    return f"{formatted}%"


def _conservative_confidence(value: float) -> float:
    clamped = float(np.clip(value, 0.0, 1.0))
    if clamped >= 1.0:
        return 1.0
    return math.floor(clamped * 100.0) / 100.0


def _should_keep_primary_tempo(
    primary_bpm: float,
    primary_confidence: float,
    ensemble_bpm: float,
) -> bool:
    if (
        primary_confidence >= 0.90
        and _relative_bpm_distance(primary_bpm, ensemble_bpm) <= 0.055
    ):
        return True
    if primary_confidence < 0.88:
        return False
    if _relative_bpm_distance(primary_bpm, ensemble_bpm) <= 0.08:
        return False
    ratio = ensemble_bpm / primary_bpm if primary_bpm > 0 else 0.0
    return _is_simple_tempo_multiple(ratio)


def _read_audio_segment(
    path: Path,
    *,
    max_analyze_seconds: float,
) -> tuple[np.ndarray, int, float, float]:
    try:
        with sf.SoundFile(str(path)) as audio_file:
            sample_rate = int(audio_file.samplerate)
            total_frames = int(len(audio_file))
            if sample_rate <= 0 or total_frames <= 0:
                raise BpmDetectionError("The file has no readable audio frames.")

            duration = total_frames / sample_rate
            analyze_seconds = min(float(max_analyze_seconds), duration)
            start_seconds = 0.0
            if duration > analyze_seconds:
                # Skip a short intro on longer tracks without jumping deep into the song.
                start_seconds = min(15.0, max(0.0, duration - analyze_seconds))

            start_frame = int(start_seconds * sample_rate)
            frames_to_read = min(
                int(analyze_seconds * sample_rate),
                total_frames - start_frame,
            )
            audio_file.seek(start_frame)
            data = audio_file.read(
                frames=frames_to_read,
                dtype="float32",
                always_2d=True,
            )
    except BpmDetectionError:
        raise
    except Exception as exc:
        raise BpmDetectionError(f"Could not read audio: {exc}") from exc

    if data.size == 0:
        raise BpmDetectionError("The selected segment has no audio data.")
    return data, sample_rate, duration, frames_to_read / sample_rate


def _read_audio_window(
    path: Path,
    *,
    start_seconds: float,
    segment_seconds: float,
) -> tuple[np.ndarray, int, float, float]:
    try:
        with sf.SoundFile(str(path)) as audio_file:
            sample_rate = int(audio_file.samplerate)
            total_frames = int(len(audio_file))
            if sample_rate <= 0 or total_frames <= 0:
                raise BpmDetectionError("The file has no readable audio frames.")

            duration = total_frames / sample_rate
            analyze_seconds = min(float(segment_seconds), duration)
            safe_start = min(max(0.0, start_seconds), max(0.0, duration - analyze_seconds))
            start_frame = int(safe_start * sample_rate)
            frames_to_read = min(
                int(analyze_seconds * sample_rate),
                total_frames - start_frame,
            )
            audio_file.seek(start_frame)
            data = audio_file.read(
                frames=frames_to_read,
                dtype="float32",
                always_2d=True,
            )
    except BpmDetectionError:
        raise
    except Exception as exc:
        raise BpmDetectionError(f"Could not read audio: {exc}") from exc

    if data.size == 0:
        raise BpmDetectionError("The selected segment has no audio data.")
    return data, sample_rate, duration, frames_to_read / sample_rate


def _prepare_mono_audio(
    samples: np.ndarray,
    *,
    source_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    mono = samples.mean(axis=1).astype(np.float32, copy=False)
    mono = np.nan_to_num(mono, copy=False)
    mono -= float(np.mean(mono))

    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak < 1e-5:
        raise BpmDetectionError("The audio is too quiet to estimate BPM.")
    mono /= peak

    if source_sample_rate != target_sample_rate:
        divisor = math.gcd(int(source_sample_rate), int(target_sample_rate))
        mono = resample_poly(
            mono,
            target_sample_rate // divisor,
            source_sample_rate // divisor,
        ).astype(np.float32, copy=False)
    return mono


def _estimate_tempo(
    samples: np.ndarray,
    *,
    sample_rate: int,
    min_bpm: float,
    max_bpm: float,
) -> tuple[float, float]:
    frame_length = 1024
    hop_length = 256
    if samples.size < frame_length * 2:
        raise BpmDetectionError("The audio is too short to estimate BPM.")

    onset_envelope = _onset_envelope(samples, sample_rate, frame_length, hop_length)
    if onset_envelope.size < 8:
        raise BpmDetectionError("The audio does not contain enough rhythmic detail.")

    frame_rate = sample_rate / hop_length
    lag_min = max(1, int(math.floor(frame_rate * 60.0 / max_bpm)))
    lag_max = min(
        onset_envelope.size - 1,
        int(math.ceil(frame_rate * 60.0 / min_bpm)),
    )
    if lag_max <= lag_min:
        raise BpmDetectionError("The audio is too short for the BPM search range.")

    envelope = onset_envelope.astype(np.float64, copy=False)
    envelope -= float(np.mean(envelope))
    envelope = np.maximum(envelope, 0.0)
    if float(np.max(envelope)) <= 1e-8:
        raise BpmDetectionError("The audio does not contain enough rhythmic detail.")

    autocorrelation = fftconvolve(envelope, envelope[::-1], mode="full")
    autocorrelation = autocorrelation[envelope.size - 1 :]
    overlap_counts = np.arange(envelope.size, 0, -1, dtype=np.float64)
    autocorrelation = autocorrelation / overlap_counts

    lags = np.arange(lag_min, lag_max + 1)
    bpms = 60.0 * frame_rate / lags
    scores = np.maximum(autocorrelation[lags], 0.0)

    # Add lighter-weight harmonic support so strong every-other-beat pulses still score well.
    for multiple, weight in ((2, 0.45), (3, 0.25)):
        harmonic_lags = lags * multiple
        valid = harmonic_lags < autocorrelation.size
        scores[valid] += weight * np.maximum(autocorrelation[harmonic_lags[valid]], 0.0)

    tempo_prior = np.exp(-0.5 * (np.log2(bpms / 120.0) / 0.9) ** 2)
    scores *= tempo_prior

    if not np.isfinite(scores).all() or float(np.max(scores)) <= 0.0:
        raise BpmDetectionError("The audio does not contain a stable beat.")

    best_index = int(np.argmax(scores))
    best_lag = float(lags[best_index])
    if 0 < best_index < scores.size - 1:
        left, center, right = scores[best_index - 1 : best_index + 2]
        denominator = left - (2.0 * center) + right
        if abs(float(denominator)) > 1e-12:
            delta = 0.5 * (left - right) / denominator
            best_lag += float(np.clip(delta, -0.5, 0.5))

    bpm = 60.0 * frame_rate / best_lag
    confidence = _confidence_from_scores(scores, best_index)
    peak_bpm, peak_confidence = _estimate_peak_interval_tempo(
        onset_envelope,
        frame_rate=frame_rate,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
    )
    if peak_bpm is not None:
        ratio = peak_bpm / bpm if bpm > 0 else 1.0
        if _relative_bpm_distance(peak_bpm, bpm) <= 0.04 and peak_confidence >= 0.65:
            bpm = peak_bpm
            confidence = max(confidence, peak_confidence)
        elif peak_confidence >= 0.65 and _is_simple_tempo_multiple(ratio):
            bpm = peak_bpm
            confidence = max(confidence, peak_confidence)
    return bpm, confidence


def _estimate_tempo_ensemble(
    path: Path,
    *,
    duration_seconds: float,
    segment_seconds: float,
    target_sample_rate: int,
    min_bpm: float,
    max_bpm: float,
) -> tuple[float, float] | None:
    deep_segment_seconds = min(
        max(20.0, float(segment_seconds)),
        DEFAULT_DEEP_WINDOW_SECONDS,
        duration_seconds,
    )
    starts = _analysis_window_starts(duration_seconds, deep_segment_seconds)
    segment_candidates: list[list[_TempoCandidate]] = []
    for start in starts:
        try:
            samples, source_sample_rate, _duration, _analyzed = _read_audio_window(
                path,
                start_seconds=start,
                segment_seconds=deep_segment_seconds,
            )
            mono = _prepare_mono_audio(
                samples,
                source_sample_rate=source_sample_rate,
                target_sample_rate=target_sample_rate,
            )
            candidates = _tempo_candidates(
                mono,
                sample_rate=target_sample_rate,
                min_bpm=min_bpm,
                max_bpm=max_bpm,
            )
        except BpmDetectionError:
            continue
        if candidates:
            segment_candidates.append(candidates)

    if len(segment_candidates) < 2:
        return None

    grid = np.linspace(min_bpm, max_bpm, int(round((max_bpm - min_bpm) * 10)) + 1)
    votes = np.zeros(grid.shape, dtype=np.float64)
    exact_weighted_sum = 0.0
    exact_weight = 0.0

    for candidates in segment_candidates:
        for rank, candidate in enumerate(candidates[:10]):
            rank_weight = 1.0 / (1.0 + (rank * 0.12))
            weight = candidate.score * rank_weight
            for variant_bpm, variant_weight in _tempo_vote_variants(
                candidate.bpm,
                min_bpm=min_bpm,
                max_bpm=max_bpm,
            ):
                _add_tempo_vote(votes, grid, variant_bpm, weight * variant_weight)

    # Prefer a human-countable metrical level when harmonic alternatives are tied.
    human_prior = np.ones_like(grid)
    human_prior[grid < 70.0] *= 0.88
    human_prior[grid > 155.0] *= 0.90
    human_prior[grid > 180.0] *= 0.78
    votes *= human_prior

    if float(np.max(votes)) <= 0.0:
        return None

    best_index = int(np.argmax(votes))
    best_bpm = float(grid[best_index])
    for candidates in segment_candidates:
        for candidate in candidates:
            if _relative_bpm_distance(candidate.bpm, best_bpm) <= 0.045:
                exact_weighted_sum += candidate.bpm * candidate.score
                exact_weight += candidate.score
    if exact_weight > 0.0:
        best_bpm = exact_weighted_sum / exact_weight

    support_scores = [
        _segment_support_strength(candidates, best_bpm)
        for candidates in segment_candidates
    ]
    supported_segments = sum(score >= 0.35 for score in support_scores)
    coverage = supported_segments / len(segment_candidates)
    support_strength = float(np.mean(support_scores)) if support_scores else 0.0
    best_score = float(votes[best_index])
    second_score = _second_independent_vote(votes, grid, best_bpm)
    separation = 1.0 if second_score <= 0.0 else max(0.0, (best_score - second_score) / best_score)
    strength = min(1.0, best_score / max(1.0, len(segment_candidates) * 2.2))
    confidence = (
        (coverage * 0.55)
        + (support_strength * 0.35)
        + (separation * 0.05)
        + (strength * 0.05)
    )
    return best_bpm, float(np.clip(confidence, 0.0, 1.0))


def _analysis_window_starts(duration_seconds: float, segment_seconds: float) -> list[float]:
    segment = min(max(20.0, float(segment_seconds)), duration_seconds)
    if duration_seconds <= segment + 5.0:
        return [0.0]

    last_start = max(0.0, duration_seconds - segment)
    raw_starts = [
        0.0,
        min(8.0, last_start),
        min(16.0, last_start),
        last_start * 0.18,
        last_start * 0.30,
        last_start * 0.42,
        last_start * 0.54,
        last_start * 0.66,
        last_start * 0.78,
        last_start * 0.90,
        max(0.0, last_start - 4.0),
        last_start,
    ]
    starts: list[float] = []
    min_spacing = max(4.0, min(8.0, segment * 0.18))
    for start in raw_starts:
        safe_start = min(max(0.0, start), last_start)
        if all(abs(safe_start - existing) >= min_spacing for existing in starts):
            starts.append(safe_start)
    return starts[:DEFAULT_DEEP_MAX_WINDOWS]


def _tempo_candidates(
    samples: np.ndarray,
    *,
    sample_rate: int,
    min_bpm: float,
    max_bpm: float,
) -> list[_TempoCandidate]:
    frame_length = 1024
    hop_length = 256
    if samples.size < frame_length * 2:
        return []

    onset_envelope = _onset_envelope(samples, sample_rate, frame_length, hop_length)
    if onset_envelope.size < 8:
        return []

    frame_rate = sample_rate / hop_length
    lag_min = max(1, int(math.floor(frame_rate * 60.0 / max_bpm)))
    lag_max = min(
        onset_envelope.size - 1,
        int(math.ceil(frame_rate * 60.0 / min_bpm)),
    )
    if lag_max <= lag_min:
        return []

    envelope = onset_envelope.astype(np.float64, copy=False)
    envelope -= float(np.mean(envelope))
    envelope = np.maximum(envelope, 0.0)
    if float(np.max(envelope)) <= 1e-8:
        return []

    autocorrelation = fftconvolve(envelope, envelope[::-1], mode="full")
    autocorrelation = autocorrelation[envelope.size - 1 :]
    autocorrelation = autocorrelation / np.arange(envelope.size, 0, -1, dtype=np.float64)

    lags = np.arange(lag_min, lag_max + 1)
    bpms = 60.0 * frame_rate / lags
    scores = np.maximum(autocorrelation[lags], 0.0)
    for multiple, weight in ((2, 0.45), (3, 0.25)):
        harmonic_lags = lags * multiple
        valid = harmonic_lags < autocorrelation.size
        scores[valid] += weight * np.maximum(autocorrelation[harmonic_lags[valid]], 0.0)
    scores *= np.exp(-0.5 * (np.log2(bpms / 120.0) / 0.9) ** 2)

    best = float(np.max(scores)) if scores.size else 0.0
    if best <= 0.0 or not np.isfinite(scores).all():
        return []

    candidates: list[_TempoCandidate] = []
    for index in np.argsort(scores)[-12:][::-1]:
        candidates.append(
            _TempoCandidate(
                bpm=float(bpms[index]),
                score=float(scores[index] / best),
                source="autocorrelation",
            )
        )

    peak_bpm, peak_confidence = _estimate_peak_interval_tempo(
        onset_envelope,
        frame_rate=frame_rate,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
    )
    if peak_bpm is not None and peak_confidence >= 0.20:
        candidates.append(
            _TempoCandidate(
                bpm=peak_bpm,
                score=min(1.25, 0.65 + peak_confidence),
                source="peak-interval",
            )
        )
    return candidates


def _tempo_vote_variants(
    bpm: float,
    *,
    min_bpm: float,
    max_bpm: float,
) -> list[tuple[float, float]]:
    variants = [
        (bpm, 1.0),
        (bpm / 2.0, 0.86 if bpm >= 145.0 else 0.68),
        (bpm * 2.0, 0.54),
        (bpm * 2.0 / 3.0, 0.72),
        (bpm * 3.0 / 2.0, 0.48),
        (bpm / 3.0, 0.38),
        (bpm * 3.0, 0.30),
    ]
    unique: list[tuple[float, float]] = []
    for variant_bpm, weight in variants:
        if min_bpm <= variant_bpm <= max_bpm and all(
            _relative_bpm_distance(variant_bpm, existing_bpm) > 0.02
            for existing_bpm, _existing_weight in unique
        ):
            unique.append((variant_bpm, weight))
    return unique


def _add_tempo_vote(
    votes: np.ndarray,
    grid: np.ndarray,
    bpm: float,
    weight: float,
) -> None:
    width = 0.045
    distance = np.log2(grid / bpm)
    votes += weight * np.exp(-0.5 * (distance / width) ** 2)


def _segment_support_strength(candidates: list[_TempoCandidate], bpm: float) -> float:
    best_support = 0.0
    for rank, candidate in enumerate(candidates[:10]):
        rank_weight = 1.0 / (1.0 + (rank * 0.12))
        for variant_bpm, variant_weight in _tempo_vote_variants(
            candidate.bpm,
            min_bpm=50.0,
            max_bpm=220.0,
        ):
            if _relative_bpm_distance(variant_bpm, bpm) <= 0.055:
                support = candidate.score * rank_weight * variant_weight
                best_support = max(best_support, support)
    return float(np.clip(best_support, 0.0, 1.0))


def _second_independent_vote(
    votes: np.ndarray,
    grid: np.ndarray,
    best_bpm: float,
) -> float:
    independent = np.asarray(
        [_is_independent_tempo(candidate_bpm, best_bpm) for candidate_bpm in grid],
        dtype=bool,
    )
    if not np.any(independent):
        return 0.0
    return float(np.max(votes[independent]))


def _estimate_peak_interval_tempo(
    envelope: np.ndarray,
    *,
    frame_rate: float,
    min_bpm: float,
    max_bpm: float,
) -> tuple[float | None, float]:
    if envelope.size < 8:
        return None, 0.0

    threshold = max(0.20, float(np.percentile(envelope, 75)))
    local_peaks = []
    for index in range(1, envelope.size - 1):
        if (
            envelope[index] >= threshold
            and envelope[index] >= envelope[index - 1]
            and envelope[index] >= envelope[index + 1]
        ):
            local_peaks.append(index)

    if len(local_peaks) < 8:
        return None, 0.0

    min_interval = frame_rate * 60.0 / max_bpm
    max_interval = frame_rate * 60.0 / min_bpm
    intervals = np.diff(np.asarray(local_peaks, dtype=np.float64))
    intervals = intervals[(intervals >= min_interval * 0.75) & (intervals <= max_interval * 1.25)]
    if intervals.size < 7:
        return None, 0.0

    median_interval = float(np.median(intervals))
    if median_interval <= 0:
        return None, 0.0

    close_intervals = intervals[
        np.abs(intervals - median_interval) <= max(1.0, median_interval * 0.16)
    ]
    if close_intervals.size < 7:
        return None, 0.0

    peak_positions = np.asarray(local_peaks, dtype=np.float64)
    steps = np.arange(peak_positions.size, dtype=np.float64)
    slope, intercept = np.polyfit(steps, peak_positions, 1)
    if slope <= 0:
        return None, 0.0

    interval = float(slope)
    bpm = 60.0 * frame_rate / interval
    while bpm < min_bpm:
        bpm *= 2.0
    while bpm > max_bpm:
        bpm /= 2.0
    if bpm < min_bpm or bpm > max_bpm:
        return None, 0.0

    fitted_positions = (slope * steps) + intercept
    variation = float(np.std(peak_positions - fitted_positions) / interval)
    coverage = close_intervals.size / max(1, intervals.size)
    confidence = coverage * max(0.0, 1.0 - (variation / 0.08))
    return bpm, float(np.clip(confidence, 0.0, 1.0))


def _relative_bpm_distance(left: float, right: float) -> float:
    if left <= 0 or right <= 0:
        return 1.0
    return abs(math.log2(left / right))


def _is_simple_tempo_multiple(ratio: float) -> bool:
    if ratio <= 0:
        return False
    return any(
        abs(math.log2(ratio / multiple)) <= 0.055
        for multiple in (0.5, 2.0, 1.0 / 3.0, 3.0)
    )


def _is_independent_tempo(candidate_bpm: float, best_bpm: float) -> bool:
    if candidate_bpm <= 0 or best_bpm <= 0:
        return True
    return not any(
        _relative_bpm_distance(candidate_bpm, best_bpm * multiple) <= 0.065
        for multiple in (1.0, 0.5, 2.0, 2.0 / 3.0, 3.0 / 2.0, 1.0 / 3.0, 3.0)
    )


def _onset_envelope(
    samples: np.ndarray,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
) -> np.ndarray:
    _, _, spectrum = stft(
        samples,
        fs=sample_rate,
        window="hann",
        nperseg=frame_length,
        noverlap=frame_length - hop_length,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(spectrum)
    if magnitude.shape[1] < 3:
        raise BpmDetectionError("The audio is too short to estimate BPM.")

    log_magnitude = np.log1p(10.0 * magnitude)
    spectral_flux = np.maximum(0.0, np.diff(log_magnitude, axis=1)).sum(axis=0)
    spectral_flux = spectral_flux.astype(np.float64, copy=False)

    smoothing_window = max(3, int(round(0.4 * sample_rate / hop_length)))
    if smoothing_window % 2 == 0:
        smoothing_window += 1
    local_average = np.convolve(
        spectral_flux,
        np.ones(smoothing_window, dtype=np.float64) / smoothing_window,
        mode="same",
    )
    envelope = np.maximum(spectral_flux - local_average, 0.0)

    peak = float(np.max(envelope)) if envelope.size else 0.0
    if peak <= 1e-9:
        raise BpmDetectionError("The audio does not contain enough rhythmic detail.")
    return envelope / peak


def _confidence_from_scores(scores: np.ndarray, best_index: int) -> float:
    peak = float(scores[best_index])
    background = float(np.percentile(scores, 60))
    if peak <= 1e-12:
        return 0.0

    return float(np.clip((peak - background) / peak, 0.0, 1.0))


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Estimate BPM for audio files.")
    parser.add_argument("files", nargs="+", help="Audio files to analyze.")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=DEFAULT_MAX_ANALYZE_SECONDS,
        help=f"Maximum seconds to analyze per file. Default: {DEFAULT_MAX_ANALYZE_SECONDS:g}",
    )
    parser.add_argument(
        "--deep-confidence",
        default=format_confidence_threshold(DEFAULT_DEEP_CONFIDENCE_THRESHOLD),
        help=(
            "Run deeper analysis when confidence is this value or lower. "
            "Accepts values like 99, 99%%, or 0.99."
        ),
    )
    args = parser.parse_args()

    exit_code = 0
    for file_name in args.files:
        try:
            result = estimate_bpm(
                file_name,
                max_analyze_seconds=args.max_seconds,
                deep_confidence_threshold=args.deep_confidence,
            )
            print(f"{file_name}: {result.bpm:.1f} BPM (confidence {result.confidence:.2f})")
        except BpmDetectionError as exc:
            exit_code = 1
            print(f"{file_name}: error: {exc}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(_main())
