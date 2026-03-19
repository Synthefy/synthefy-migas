"""
ENTSO-E Transparency Platform — Bulk SFTP Download
===================================================
Downloads all CSV data (zipped) from the ENTSO-E SFTP server.

Usage:
    python entsoe_bulk_download.py

You'll be prompted for your Transparency Platform credentials
(the same email/password you use to log in to https://transparency.entsoe.eu).

Data lands in ./entsoe_data/ organized by data-item folder.
"""

import os
import sys
import stat
import getpass
import argparse
from pathlib import Path

try:
    import paramiko
except ImportError:
    sys.exit(
        "paramiko is required.\n"
        "Install it with:  pip install paramiko"
    )


# ── Connection details ──────────────────────────────────────────────────────
SFTP_HOST = "sftp-transparency.entsoe.eu"
SFTP_PORT = 22
REMOTE_ROOT = "/TP_export"
REMOTE_ZIP = "/TP_export/zip"


def connect(username: str, password: str) -> paramiko.SFTPClient:
    """Open an SFTP connection and return the client."""
    transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    print(f"Connected to {SFTP_HOST}")
    return sftp


def list_remote_dirs(sftp: paramiko.SFTPClient, remote_path: str) -> list[str]:
    """Return subfolder names under remote_path."""
    dirs = []
    for entry in sftp.listdir_attr(remote_path):
        if stat.S_ISDIR(entry.st_mode):
            dirs.append(entry.filename)
    return sorted(dirs)


def list_remote_files(sftp: paramiko.SFTPClient, remote_path: str) -> list[str]:
    """Return file names (non-dirs) under remote_path."""
    files = []
    for entry in sftp.listdir_attr(remote_path):
        if not stat.S_ISDIR(entry.st_mode):
            files.append(entry.filename)
    return sorted(files)


def download_folder(
    sftp: paramiko.SFTPClient,
    remote_dir: str,
    local_dir: Path,
    use_zip: bool = True,
    skip_existing: bool = True,
):
    """Download every file in a single remote folder."""
    local_dir.mkdir(parents=True, exist_ok=True)
    files = list_remote_files(sftp, remote_dir)
    if not files:
        return 0

    downloaded = 0
    for fname in files:
        remote_path = f"{remote_dir}/{fname}"
        local_path = local_dir / fname

        if skip_existing and local_path.exists():
            continue

        try:
            sftp.get(remote_path, str(local_path))
            downloaded += 1
        except Exception as e:
            print(f"  !! Failed: {fname} — {e}")

    return downloaded


def download_all(
    sftp: paramiko.SFTPClient,
    local_root: Path,
    use_zip: bool = True,
    skip_existing: bool = True,
):
    """
    Walk the SFTP tree and download everything.

    If use_zip=True, pulls from /TP_export/zip/ (much faster).
    Otherwise pulls raw CSVs from /TP_export/<folder>/.
    """
    base = REMOTE_ZIP if use_zip else REMOTE_ROOT

    # Top-level files (export log, etc.)
    top_files = list_remote_files(sftp, base)
    if top_files:
        print(f"\nDownloading {len(top_files)} top-level file(s)...")
        download_folder(sftp, base, local_root, use_zip, skip_existing)

    # Data-item subfolders
    folders = list_remote_dirs(sftp, base)
    total_folders = len(folders)
    print(f"\nFound {total_folders} data-item folder(s) under {base}/\n")

    grand_total = 0
    for i, folder_name in enumerate(folders, 1):
        remote_path = f"{base}/{folder_name}"
        local_path = local_root / folder_name

        files = list_remote_files(sftp, remote_path)
        print(f"[{i}/{total_folders}] {folder_name}  ({len(files)} files)")

        n = download_folder(sftp, remote_path, local_path, use_zip, skip_existing)
        grand_total += n
        if n:
            print(f"         ↳ downloaded {n} new file(s)")

    print(f"\nDone. {grand_total} file(s) downloaded to {local_root.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Bulk-download ENTSO-E Transparency Platform data via SFTP."
    )
    parser.add_argument(
        "-o", "--output",
        default="./entsoe_data",
        help="Local directory to save data (default: ./entsoe_data)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Download raw (uncompressed) CSVs instead of zipped versions.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    parser.add_argument(
        "-u", "--username",
        default=None,
        help="ENTSO-E registered email (prompted if omitted).",
    )
    args = parser.parse_args()

    # ── Credentials (env vars > CLI arg > interactive prompt) ────────────
    username = args.username or os.environ.get("ENTSOE_USER") or input("ENTSO-E email: ").strip()
    password = os.environ.get("ENTSOE_PASSWORD") or getpass.getpass("ENTSO-E password: ")

    # ── Connect & download ───────────────────────────────────────────────
    sftp = connect(username, password)
    try:
        download_all(
            sftp,
            local_root=Path(args.output),
            use_zip=not args.raw,
            skip_existing=not args.force,
        )
    finally:
        sftp.close()


if __name__ == "__main__":
    main()
