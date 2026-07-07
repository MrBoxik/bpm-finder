from __future__ import annotations

import base64
import ctypes
import os
import queue
import struct
import sys
import threading
import time
import tkinter as tk
from concurrent.futures._base import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from audio_formats import AUDIO_FILE_DIALOG_PATTERN, is_audio_file
from bpm_finder import (
    DEFAULT_DEEP_CONFIDENCE_THRESHOLD,
    DEFAULT_MAX_ANALYZE_SECONDS,
    BpmDetectionError,
    estimate_bpm,
    format_confidence_threshold,
    parse_confidence_threshold,
)
from exporters import write_xlsx


BUY_ME_A_COFFEE_URL = "https://buymeacoffee.com/mrboxik"
TECH_SUPPORT_URL = "https://discord.com/users/638802769393745950"
GITHUB_URL = "https://github.com/MrBoxik/bpm-finder"
LICENSE_URL = "https://github.com/MrBoxik/bpm-finder/blob/main/LICENSE.txt"
APP_USER_MODEL_ID = "MrBoxik.BPMFinder"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _default_worker_count() -> int:
    logical_cpus = os.cpu_count() or 1
    if logical_cpus <= 2:
        return logical_cpus
    return max(2, min(12, round(logical_cpus * 0.75)))


def _resource_path(relative_path: str) -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")) / relative_path
    return Path(__file__).resolve().parent.parent / relative_path


def _set_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except (AttributeError, OSError):
        pass


def _ico_png_layers(path: Path) -> list[tuple[int, bytes]]:
    try:
        data = path.read_bytes()
    except OSError:
        return []

    if len(data) < 6:
        return []

    try:
        reserved, icon_type, count = struct.unpack_from("<HHH", data, 0)
    except struct.error:
        return []

    if reserved != 0 or icon_type != 1 or count <= 0:
        return []

    layers: list[tuple[int, bytes]] = []
    for index in range(count):
        entry_offset = 6 + index * 16
        if entry_offset + 16 > len(data):
            break

        width, _height, _colors, _reserved_b, _planes, _bit_count, size, image_offset = (
            struct.unpack_from("<BBBBHHII", data, entry_offset)
        )
        icon_size = 256 if width == 0 else width
        image_end = image_offset + size
        if image_offset >= len(data) or image_end > len(data):
            continue

        image = data[image_offset:image_end]
        if image.startswith(PNG_SIGNATURE):
            layers.append((icon_size, image))

    layers.sort(key=lambda item: item[0])
    return layers


DEFAULT_WORKERS = _default_worker_count()
MAX_UI_MESSAGES_PER_POLL = 250


class BpmFinderApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("BPM Finder")
        self.root.geometry("840x600")
        self.root.minsize(720, 500)
        self.icon_images: list[tk.PhotoImage] = []
        self._set_window_icon()

        self.results_queue: queue.Queue[tuple] = queue.Queue()
        self.path_to_item: dict[Path, str] = {}
        self.export_records: dict[Path, dict[str, str]] = {}
        self.worker: threading.Thread | None = None
        self.total_jobs = 0
        self.finished_jobs = 0
        self.analysis_started_at = 0.0

        self._build_ui()
        self.root.after(100, self._poll_results)

    def _set_window_icon(self) -> None:
        icon_path = _resource_path("icon.ico")
        if not icon_path.exists():
            return

        self.icon_images = []
        for _size, image in _ico_png_layers(icon_path):
            try:
                encoded = base64.b64encode(image).decode("ascii")
                self.icon_images.append(tk.PhotoImage(data=encoded))
            except tk.TclError:
                continue

        if self.icon_images:
            try:
                self.root.iconphoto(True, *self.icon_images)
                return
            except tk.TclError:
                pass

        try:
            self.root.iconbitmap(str(icon_path))
        except tk.TclError:
            pass

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        self.status_var = tk.StringVar(value="Add audio files or a folder to begin.")

        toolbar = ttk.Frame(self.root, padding=(10, 10, 10, 6))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(0, weight=1)

        actions = ttk.Frame(toolbar)
        actions.grid(row=0, column=0, sticky="w")

        self.add_button = ttk.Button(actions, text="Add Files", command=self.add_files)
        self.add_button.grid(row=0, column=0, padx=(0, 6))

        self.folder_button = ttk.Button(actions, text="Add Folder", command=self.add_folder)
        self.folder_button.grid(row=0, column=1, padx=(0, 6))

        self.clear_button = ttk.Button(actions, text="Clear All", command=self.clear_files)
        self.clear_button.grid(row=0, column=2, padx=(0, 14))

        self.find_button = ttk.Button(actions, text="Find BPM", command=self.analyze_files)
        self.find_button.grid(row=0, column=3, padx=(0, 6))

        self.save_excel_button = ttk.Button(
            actions,
            text="Save in Excel",
            command=self.save_excel,
        )
        self.save_excel_button.grid(row=0, column=4, padx=(0, 6))

        status_label = ttk.Label(toolbar, textvariable=self.status_var, anchor="w")
        status_label.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        settings = ttk.LabelFrame(self.root, text="Settings", padding=(10, 6, 10, 8))
        settings.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))
        settings.columnconfigure(7, weight=1)

        ttk.Label(settings, text="CPU Workers").grid(row=0, column=0, padx=(0, 4))
        self.worker_count_var = tk.IntVar(value=DEFAULT_WORKERS)
        self.worker_spinbox = ttk.Spinbox(
            settings,
            from_=1,
            to=max(1, min(24, os.cpu_count() or 1)),
            width=4,
            textvariable=self.worker_count_var,
        )
        self.worker_spinbox.grid(row=0, column=1, padx=(0, 16))

        ttk.Label(settings, text="Seconds").grid(row=0, column=2, padx=(0, 4))
        self.max_seconds_var = tk.IntVar(value=int(DEFAULT_MAX_ANALYZE_SECONDS))
        self.max_seconds_spinbox = ttk.Spinbox(
            settings,
            from_=20,
            to=300,
            increment=5,
            width=5,
            textvariable=self.max_seconds_var,
        )
        self.max_seconds_spinbox.grid(row=0, column=3, padx=(0, 16))

        ttk.Label(settings, text="Deep Check Below").grid(row=0, column=4, padx=(0, 4))
        self.deep_confidence_var = tk.StringVar(
            value=format_confidence_threshold(DEFAULT_DEEP_CONFIDENCE_THRESHOLD)
        )
        self.deep_confidence_entry = ttk.Entry(
            settings,
            width=7,
            textvariable=self.deep_confidence_var,
        )
        self.deep_confidence_entry.grid(row=0, column=5, padx=(0, 16))

        logical_cpus = os.cpu_count() or 1
        settings_help = (
            f"Auto selected {DEFAULT_WORKERS} worker(s) from {logical_cpus} logical CPU "
            "thread(s). CPU workers analyze multiple files at the same time; more can be "
            "faster until your CPU is busy. Seconds is the maximum audio scanned per file; "
            "lower is faster, higher can help tracks with long intros or unstable beats. "
            "Deep Check Below runs the heavier checker when confidence is at or under that "
        )
        ttk.Label(settings, text=settings_help, wraplength=780, justify="left").grid(
            row=1,
            column=0,
            columnspan=8,
            sticky="ew",
            pady=(6, 0),
        )

        table_frame = ttk.Frame(self.root, padding=(10, 0, 10, 6))
        table_frame.grid(row=2, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        columns = ("file", "bpm", "confidence", "status")
        self.tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            selectmode="extended",
        )
        self.tree.heading("file", text="File")
        self.tree.heading("bpm", text="BPM")
        self.tree.heading("confidence", text="Confidence")
        self.tree.heading("status", text="Status")
        self.tree.column("file", width=330, minwidth=220, anchor="w", stretch=True)
        self.tree.column("bpm", width=70, minwidth=65, anchor="center", stretch=False)
        self.tree.column("confidence", width=95, minwidth=90, anchor="center", stretch=False)
        self.tree.column("status", width=180, minwidth=140, anchor="w", stretch=True)
        self.tree.grid(row=0, column=0, sticky="nsew")

        y_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=y_scroll.set)

        bottom = ttk.Frame(self.root, padding=(10, 0, 10, 6))
        bottom.grid(row=3, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)

        self.progress = ttk.Progressbar(bottom, mode="determinate")
        self.progress.grid(row=0, column=0, sticky="ew")

        links = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        links.grid(row=4, column=0, sticky="ew")

        ttk.Button(
            links,
            text="Buy Me a Coffee",
            command=lambda: self._open_url(BUY_ME_A_COFFEE_URL),
        ).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(
            links,
            text="Tech Support",
            command=lambda: self._open_url(TECH_SUPPORT_URL),
        ).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(
            links,
            text="GitHub Page",
            command=lambda: self._open_url(GITHUB_URL),
        ).grid(row=0, column=2, padx=(0, 6))
        ttk.Button(
            links,
            text="License",
            command=lambda: self._open_url(LICENSE_URL),
        ).grid(row=0, column=3, padx=(0, 6))

    def _open_url(self, url: str) -> None:
        try:
            os.startfile(url)
        except OSError:
            messagebox.showerror("BPM Finder", f"Could not open link:\n{url}")

    def add_files(self) -> None:
        file_names = filedialog.askopenfilenames(
            title="Choose audio files",
            filetypes=(("Audio files", AUDIO_FILE_DIALOG_PATTERN), ("All files", "*.*")),
        )
        self._add_paths(Path(name) for name in file_names)

    def add_folder(self) -> None:
        folder_name = filedialog.askdirectory(title="Choose a folder containing audio files")
        if not folder_name:
            return
        folder = Path(folder_name)
        paths = sorted(
            (path for path in folder.rglob("*") if is_audio_file(path)),
            key=lambda path: str(path).casefold(),
        )
        self._add_paths(paths)

    def _add_paths(self, paths) -> None:
        added = 0
        for raw_path in paths:
            path = Path(raw_path).resolve()
            if not is_audio_file(path) or path in self.path_to_item:
                continue
            item_id = self.tree.insert(
                "",
                "end",
                values=(path.name, "", "", "Pending"),
            )
            self.path_to_item[path] = item_id
            self.export_records[path] = {
                "file": path.name,
                "bpm": "",
                "confidence": "",
                "status": "Pending",
            }
            added += 1
        if added:
            self.status_var.set(f"Added {added} file(s). Total: {len(self.path_to_item)}.")
        else:
            self.status_var.set("No new supported audio files found.")

    def clear_files(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("BPM Finder", "Wait for analysis to finish before clearing files.")
            return
        for item_id in self.tree.get_children():
            self.tree.delete(item_id)
        self.path_to_item.clear()
        self.export_records.clear()
        self.progress["value"] = 0
        self.status_var.set("Cleared.")

    def analyze_files(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        paths = list(self.path_to_item.keys())
        if not paths:
            messagebox.showinfo("BPM Finder", "Add at least one audio file first.")
            return

        workers = self._validated_worker_count()
        max_seconds = self._validated_max_seconds()
        try:
            deep_confidence_threshold = self._validated_deep_confidence_threshold()
        except ValueError as exc:
            messagebox.showerror("BPM Finder", str(exc))
            return

        self.total_jobs = len(paths)
        self.finished_jobs = 0
        self.analysis_started_at = time.perf_counter()
        self.progress["maximum"] = self.total_jobs
        self.progress["value"] = 0
        self._set_controls_enabled(False)
        for path in paths:
            self._update_row(path, bpm="", confidence="", status="Queued")

        self.worker = threading.Thread(
            target=self._worker_analyze,
            args=(paths, workers, max_seconds, deep_confidence_threshold),
            daemon=True,
        )
        self.worker.start()
        self.status_var.set(
            f"Finding BPM for {self.total_jobs} file(s) with {workers} worker(s)..."
        )

    def _worker_analyze(
        self,
        paths: list[Path],
        workers: int,
        max_seconds: float,
        deep_confidence_threshold: float,
    ) -> None:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self._analyze_one_file,
                    path,
                    max_seconds,
                    deep_confidence_threshold,
                ): path
                for path in paths
            }
            for future in as_completed(futures):
                self.results_queue.put(future.result())
        self.results_queue.put(("done",))

    def _analyze_one_file(
        self,
        path: Path,
        max_seconds: float,
        deep_confidence_threshold: float,
    ) -> tuple:
        self.results_queue.put(("status", path, "Finding BPM"))
        started = time.perf_counter()
        try:
            result = estimate_bpm(
                path,
                max_analyze_seconds=max_seconds,
                deep_confidence_threshold=deep_confidence_threshold,
            )
            return (
                "result",
                path,
                result.bpm,
                result.confidence,
                time.perf_counter() - started,
            )
        except BpmDetectionError as exc:
            return ("error", path, str(exc))
        except Exception as exc:
            return ("error", path, f"Unexpected error: {exc}")

    def _poll_results(self) -> None:
        processed = 0
        try:
            while processed < MAX_UI_MESSAGES_PER_POLL:
                processed += 1
                message = self.results_queue.get_nowait()
                kind = message[0]
                if kind == "status":
                    _, path, status = message
                    self._update_row(path, status=status)
                elif kind == "result":
                    _, path, bpm, confidence, elapsed = message
                    self.finished_jobs += 1
                    self.progress["value"] = self.finished_jobs
                    self._update_row(
                        path,
                        bpm=f"{bpm:.1f}",
                        confidence=self._format_confidence(confidence),
                        status=f"Done ({elapsed:.1f}s)",
                    )
                    self._update_analysis_status()
                elif kind == "error":
                    _, path, error = message
                    self.finished_jobs += 1
                    self.progress["value"] = self.finished_jobs
                    self._update_row(path, bpm="", confidence="", status=error)
                    self._update_analysis_status()
                elif kind == "done":
                    self._set_controls_enabled(True)
                    elapsed = max(0.01, time.perf_counter() - self.analysis_started_at)
                    rate = self.finished_jobs / elapsed
                    self.status_var.set(
                        f"Finished {self.finished_jobs} of {self.total_jobs} file(s) "
                        f"in {elapsed:.1f}s ({rate:.1f} files/sec)."
                    )
        except queue.Empty:
            pass
        self.root.after(100, self._poll_results)

    def _update_analysis_status(self) -> None:
        if not self.analysis_started_at:
            return
        elapsed = max(0.01, time.perf_counter() - self.analysis_started_at)
        rate = self.finished_jobs / elapsed
        self.status_var.set(
            f"Finding BPM... {self.finished_jobs}/{self.total_jobs} "
            f"({rate:.1f} files/sec)"
        )

    def _validated_worker_count(self) -> int:
        max_workers = max(1, min(24, os.cpu_count() or 1))
        try:
            value = int(self.worker_count_var.get())
        except tk.TclError:
            value = DEFAULT_WORKERS
        return max(1, min(max_workers, value))

    def _validated_max_seconds(self) -> float:
        try:
            value = float(self.max_seconds_var.get())
        except tk.TclError:
            value = DEFAULT_MAX_ANALYZE_SECONDS
        validated = max(20.0, min(300.0, value))
        self.max_seconds_var.set(int(validated))
        return validated

    def _validated_deep_confidence_threshold(self) -> float:
        threshold = parse_confidence_threshold(self.deep_confidence_var.get())
        self.deep_confidence_var.set(format_confidence_threshold(threshold))
        return threshold

    def _format_confidence(self, confidence: float) -> str:
        clamped = max(0.0, min(1.0, confidence))
        return f"{int(clamped * 100)}%"

    def _update_row(
        self,
        path: Path,
        *,
        bpm: str | None = None,
        confidence: str | None = None,
        status: str | None = None,
    ) -> None:
        item_id = self.path_to_item.get(path)
        if not item_id:
            return
        record = self.export_records.setdefault(
            path,
            {
                "file": path.name,
                "bpm": "",
                "confidence": "",
                "status": "",
            },
        )
        if bpm is not None:
            record["bpm"] = bpm
        if confidence is not None:
            record["confidence"] = confidence
        if status is not None:
            record["status"] = status
        self.tree.item(
            item_id,
            values=(
                record["file"],
                record["bpm"],
                record["confidence"],
                record["status"],
            ),
        )

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for control in (
            self.add_button,
            self.folder_button,
            self.clear_button,
            self.find_button,
            self.save_excel_button,
            self.worker_spinbox,
            self.max_seconds_spinbox,
            self.deep_confidence_entry,
        ):
            control.configure(state=state)

    def save_excel(self) -> None:
        if not self.path_to_item:
            messagebox.showinfo("BPM Finder", "There are no results to save.")
            return
        file_name = filedialog.asksaveasfilename(
            title="Save BPM results in Excel",
            defaultextension=".xlsx",
            filetypes=(("Excel workbook", "*.xlsx"), ("All files", "*.*")),
        )
        if not file_name:
            return

        headers, rows = self._export_rows()
        write_xlsx(file_name, headers, rows)
        self.status_var.set(f"Saved {file_name}")

    def _export_rows(self) -> tuple[list[str], list[tuple[str, ...]]]:
        headers = ["file", "bpm", "confidence", "status"]
        rows = [
            (
                self.export_records[path]["file"],
                self.export_records[path]["bpm"],
                self.export_records[path]["confidence"],
                self.export_records[path]["status"],
            )
            for path in self.path_to_item
        ]
        return headers, rows


def main() -> None:
    _set_windows_app_id()
    root = tk.Tk()
    BpmFinderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
