# Releasing Standalone Binaries

This repository is configured to build standalone desktop app artifacts with GitHub Actions and publish them to GitHub Releases.

## What gets published

On each version tag (`v*`), the workflow builds and uploads:

- `talk-to-text-windows-x64.zip`
- `talk-to-text-macos-x64.zip`
- `talk-to-text-macos-arm64.zip`

Each archive contains a self-contained PyInstaller app package, so end users do not need Python installed.

- macOS archives contain `Talk to Text.app`
- Windows archives contain the `Talk to Text/` app directory (with `Talk to Text.exe` and bundled runtime files)

## How to release

1. Commit your changes to the default branch.
2. Create a version tag:
   - `git tag v1.0.0`
3. Push the tag:
   - `git push origin v1.0.0`

GitHub Actions workflow:

- `.github/workflows/release-binaries.yml`

It will create/update the GitHub Release for that tag and attach the built archives.

## Local build (same packaging model)

Install local build tools:

```bash
python -m pip install --upgrade pip
pip install pipenv
pipenv install --system --deploy
pip install pyinstaller
```

Build app package:

```bash
pyinstaller --noconfirm --clean --windowed --name "Talk to Text" transcriber/ui_app.py
```

Output:

- macOS: `dist/Talk to Text.app`
- Windows: `dist/Talk to Text/` (contains `Talk to Text.exe`)

## Runtime note

The app binary includes Python runtime and Python dependencies. On first run, the app may still download external transcription assets/tools (for example, ffmpeg/whisper.cpp/model files) based on current app behavior.
