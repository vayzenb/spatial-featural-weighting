#!/usr/bin/env python3
"""
Delete images from a folder that are NOT referenced in a COCO-style JSON file.

The JSON must contain:
{
    "images": [
        {"id": ..., "file_name": "...", "width": ..., "height": ...},
        ...
    ],
    "annotations": [...],
    "categories": [...]
}

Usage:
    python delete_unused_images.py path/to/data.json path/to/image_folder
"""

import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Delete images not listed in a COCO-style JSON file.")
    parser.add_argument("json_path", type=Path, help="Path to the JSON file")
    parser.add_argument("image_folder", type=Path, help="Folder containing the images to clean")
    args = parser.parse_args()

    # ---- Load JSON ----
    print(f"[info] Loading JSON: {args.json_path}")
    with args.json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Get allowed filenames from JSON
    json_images = data.get("images", [])
    keep_filenames = set(img["file_name"] for img in json_images)

    print(f"[info] JSON lists {len(keep_filenames)} images to keep.")

    # ---- Process Folder ----
    image_dir = args.image_folder
    if not image_dir.exists():
        print(f"[error] Folder does not exist: {image_dir}")
        return

    removed = 0
    kept = 0

    # Loop over all files in the folder
    for file in image_dir.iterdir():
        if file.is_file():
            if file.name not in keep_filenames:
                print(f"[delete] Removing: {file.name}")
                file.unlink()  # delete the file
                removed += 1
            else:
                kept += 1

    print(f"\n[done] Finished cleaning folder.")
    print(f"Images kept:    {kept}")
    print(f"Images removed: {removed}")
    print(f"Total files seen: {kept + removed}")


if __name__ == "__main__":
    main()
