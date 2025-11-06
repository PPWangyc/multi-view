#!/usr/bin/env python3

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_URL = "https://figshare.com/ndownloader/files/39698587"
DEFAULT_FILENAME = "sebea_dataset.zip"


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0


def download_with_progress(url: str, dest_path: Path) -> None:
    def reporthook(block_num: int, block_size: int, total_size: int):
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100.0, downloaded * 100.0 / total_size)
            downloaded = min(downloaded, total_size)
            msg = f"\rDownloading: {format_bytes(downloaded)} / {format_bytes(total_size)} ({percent:5.1f}%)"
        else:
            downloaded = block_num * block_size
            msg = f"\rDownloading: {format_bytes(downloaded)}"
        sys.stdout.write(msg)
        sys.stdout.flush()

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=reporthook)
        sys.stdout.write("\n")
        tmp_path.replace(dest_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)


def find_external_extractor(preferred: str | None = None) -> tuple[list[str], str]:
    # Returns (command_list, tool_name) or ([], "") if none found
    candidates: list[tuple[list[str], str]] = []
    if preferred in (None, "7z"):
        sevenz = shutil.which("7z") or shutil.which("7za")
        if sevenz:
            candidates.append(([sevenz, "x", "-y"], "7z"))
    if preferred in (None, "bsdtar"):
        bsdtar = shutil.which("bsdtar")
        if bsdtar:
            candidates.append(([bsdtar, "-xvf"], "bsdtar"))
    return candidates[0] if candidates else ([], "")


def extract_with_external(zip_path: Path, extract_dir: Path, tool: str | None = None) -> None:
    cmd_base, tool_name = find_external_extractor(preferred=tool)
    if not cmd_base:
        raise RuntimeError("No external extractor found (7z or bsdtar). Install p7zip or bsdtar.")
    extract_dir.mkdir(parents=True, exist_ok=True)
    if tool_name == "7z":
        cmd = cmd_base + [f"-o{str(extract_dir)}", str(zip_path)]
    elif tool_name == "bsdtar":
        cmd = cmd_base + [str(zip_path), "-C", str(extract_dir)]
    else:
        raise RuntimeError("Unsupported external tool")
    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"External extractor {tool_name} failed with code {proc.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract the SEBEA dataset zip from Figshare.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Figshare direct download URL")
    parser.add_argument("--output-dir", default=str(Path("data") / "sebea"), help="Directory to save and extract the dataset")
    parser.add_argument("--filename", default=DEFAULT_FILENAME, help="Filename for the downloaded zip")
    parser.add_argument("--skip-extract", action="store_true", help="Only download the zip, do not extract")
    parser.add_argument("--extractor", choices=["auto", "zipfile", "7z", "bsdtar"], default="auto", help="Extractor to use for zip contents")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / args.filename

    # If zip already exists and is non-empty, skip download
    if zip_path.exists() and zip_path.stat().st_size > 0:
        print(f"Zip already exists at: {zip_path} ({format_bytes(zip_path.stat().st_size)}). Skipping download.")
    else:
        print(f"Starting download from: {args.url}")
        try:
            download_with_progress(args.url, zip_path)
        except urllib.error.HTTPError as e:
            print(f"HTTP error: {e.code} {e.reason}")
            sys.exit(1)
        except urllib.error.URLError as e:
            print(f"URL error: {e.reason}")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nDownload interrupted by user.")
            # Clean up partial file if present
            if zip_path.exists():
                try:
                    zip_path.unlink()
                except Exception:
                    pass
            sys.exit(130)
        except Exception as e:
            print(f"Unexpected error during download: {e}")
            sys.exit(1)

    if args.skip_extract:
        print(f"Downloaded zip saved to: {zip_path}")
        return

    # Extract
    extract_dir = output_dir / "extracted"
    if extract_dir.exists():
        # If previously extracted, keep as-is
        print(f"Extraction directory already exists: {extract_dir}. Skipping extraction.")
        print(f"Zip is at: {zip_path}")
        return

    print(f"Extracting to: {extract_dir}")
    extract_dir.mkdir(parents=True, exist_ok=True)

    def try_zipfile() -> bool:
        try:
            extract_zip(zip_path, extract_dir)
            return True
        except zipfile.BadZipFile:
            print("The downloaded file is not a valid zip. Delete it and try again.")
            sys.exit(1)
        except NotImplementedError as nie:
            # e.g., Deflate64 not supported
            print(f"zipfile cannot extract this archive: {nie}")
            return False
        except Exception as e:
            print(f"Unexpected error during extraction with zipfile: {e}")
            return False

    if args.extractor == "zipfile":
        if not try_zipfile():
            sys.exit(1)
    elif args.extractor in ("7z", "bsdtar"):
        try:
            extract_with_external(zip_path, extract_dir, tool=args.extractor)
        except Exception as e:
            print(f"External extraction failed: {e}")
            sys.exit(1)
    else:  # auto
        if not try_zipfile():
            try:
                extract_with_external(zip_path, extract_dir, tool="7z")
            except Exception:
                try:
                    extract_with_external(zip_path, extract_dir, tool="bsdtar")
                except Exception as e:
                    print(f"All extraction methods failed: {e}")
                    print("Hint: install 7-Zip (p7zip/p7zip-plugins) or bsdtar and retry, or run with --extractor 7z")
                    sys.exit(1)

    print("Done.")
    print(f"Zip: {zip_path}")
    print(f"Extracted contents: {extract_dir}")


if __name__ == "__main__":
    main()
