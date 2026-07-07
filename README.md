# BPM Finder

A simple Windows app that finds the BPM of music and audio files.

It is built for large folders, keeps the results readable in the app, and saves a clean Excel workbook with proper Unicode filenames.

---

## Download

<p align="center">

<a href="https://github.com/MrBoxik/bpm-finder/releases" style="font-size:20px;">
  <b>Download for Windows</b>
</a>

</p>

---

## Features

- Find BPM for audio files and whole music folders
- Supports common audio formats like MP3, WAV, FLAC, OGG, Opus, AIFF, M4A, AAC, and WMA when the decoder can read them
- Fast CPU worker setting for big libraries
- Seconds setting to control how much audio is scanned per file
- Confidence score for each result
- Clean Excel export with File, BPM, Confidence, and Status columns

---

## How To Use

1. Download the Windows release.
2. Open `BPMFinder.exe`.
3. Click **Add Files** or **Add Folder**.
4. Adjust **CPU Workers** if you want more or less parallel processing.
5. Adjust **Seconds** if you want faster scans or more careful scans.
6. Click **Find BPM**.
7. Click **Save in Excel** to export the results.

---

## Settings

**CPU Workers** controls how many files are analyzed at the same time.

The app auto-picks a good default based on your logical CPU thread count. More workers can be faster for large folders, but setting it too high can make the computer feel busy.

**Seconds** controls the maximum amount of audio scanned per file.

Lower values are faster. Higher values can help tracks with long intros, quiet sections, or unstable rhythm.

---

## 💬 Feedback or Questions?

You can leave feedback [here on GitHub](https://github.com/MrBoxik/SnowRunner-Save-Editor/issues) — but I’m not going to be super active here.  

You’ll have better chances of getting a reply if you message me on [Discord](https://discord.com/users/638802769393745950).

---

## ☕ Support

If this save editor helped you and you want to say thanks, you can [buy me a coffee](https://buymeacoffee.com/mrboxik).  
Totally optional, just appreciated. ❤️


---

## 📜 License

This project is licensed under a **Custom Non-Commercial License**.  

You are free to use, modify, and share the code for personal and non-commercial purposes, with attribution.  
**Commercial use is not allowed** without prior written permission from the author.  

See the [LICENSE](LICENSE) file for full details.  
