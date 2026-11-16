"""
Archives the entire completed project into a clean ZIP bundle for delivery and evaluation.
"""

import os
import zipfile
from pathlib import Path

BASE_DIR = Path(".")
ZIP_NAME = "ScholarAI_MCA_NLP_Final_Project.zip"
EXCLUDE_DIRS = {"__pycache__", ".git", ".idea", ".vscode", "venv", ".pytest_cache"}
EXCLUDE_EXTS = {".pyc", ".pyo", ".tmp"}


def make_project_zip():
    zip_path = BASE_DIR / ZIP_NAME
    total_files = 0
    total_bytes = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(BASE_DIR):
            # Modify dirs in-place to skip excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in files:
                if file == ZIP_NAME:
                    continue
                file_path = Path(root) / file
                if file_path.suffix in EXCLUDE_EXTS:
                    continue

                rel_path = file_path.relative_to(BASE_DIR)
                zf.write(file_path, arcname=str(rel_path))
                total_files += 1
                total_bytes += file_path.stat().st_size

    size_mb = round(zip_path.stat().st_size / (1024 * 1024), 2)
    print(f"Successfully packaged {total_files} files ({round(total_bytes/(1024*1024), 2)} MB uncompressed) into {zip_path.resolve()} ({size_mb} MB compressed).")


if __name__ == "__main__":
    make_project_zip()

