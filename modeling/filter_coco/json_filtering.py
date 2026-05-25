#!/usr/bin/env python3
"""
Filter COCO instances JSON based on:

  1) Remove *images* that have ANY annotation with iscrowd == 1,
     and remove ALL annotations belonging to those images.

  2) Remove annotations whose bbox area is < MIN_FRACTION_OF_IMAGE * image area

  3) EXCLUSION CATEGORIES (UPDATED BEHAVIOR):
     If an image has ANY annotation whose category *name* is in EXCLUDED_CATEGORY_NAMES,
     remove the ENTIRE image and ALL its annotations.

  4) After the above, remove any image that still has > MAX_BBOX_PER_IMAGE annotations
     (and remove those annotations too)

  5) Remove any image that ends up with zero remaining annotations (implicit)

Keeps the original 'categories' unchanged.

Usage:
  python json_filtering_drop_excluded_images.py \
      /path/to/instances_train2017.json \
      /path/to/instances_train2017_filtered.json
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


# -----------------------------
# CONFIG
# -----------------------------

MIN_FRACTION_OF_IMAGE = 1.0 / 16.0
MAX_BBOX_PER_IMAGE = 5

EXCLUDED_CATEGORY_NAMES = [
    "person",
    "tie",
    "skis",
    "snowboard",
    "sink",
    "train",
    "apple",
    "banana",
    "broccoli",
    "orange",
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    # If you want pretty output, add: indent=2
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def build_image_lookup(images):
    """Return dict: image_id -> image_dict"""
    return {img["id"]: img for img in images}


def build_category_lookup(categories):
    """Return dict: category_id -> category_name_lower"""
    out = {}
    for cat in categories:
        cid = cat.get("id")
        name = (cat.get("name") or "").strip().lower()
        if cid is not None:
            out[cid] = name
    return out


def filter_coco(data):
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    excluded_names_set = {name.strip().lower() for name in EXCLUDED_CATEGORY_NAMES}
    cat_id_to_name = build_category_lookup(categories)

    # -------- 1) Find images that have ANY crowd annotation --------
    crowd_image_ids = {
        ann.get("image_id")
        for ann in annotations
        if ann.get("iscrowd", 0) == 1 and ann.get("image_id") is not None
    }

    # -------- 3) Find images that have ANY excluded-category annotation --------
    # (Do this on the original annotations so we drop the whole image early.)
    excluded_image_ids = set()
    for ann in annotations:
        img_id = ann.get("image_id")
        if img_id is None:
            continue
        cat_id = ann.get("category_id")
        cat_name = cat_id_to_name.get(cat_id, "")
        if cat_name in excluded_names_set:
            excluded_image_ids.add(img_id)

    # Images to drop for "whole-image" reasons:
    drop_image_ids = set(crowd_image_ids) | set(excluded_image_ids)

    # Remove annotations belonging to dropped images
    annotations_after_drop = [
        ann for ann in annotations
        if ann.get("image_id") not in drop_image_ids
    ]

    # Remove images themselves
    images_after_drop = [
        img for img in images
        if img.get("id") not in drop_image_ids
    ]

    images_by_id = build_image_lookup(images_after_drop)

    # -------- 2) Remove too-small boxes (annotation-level) --------
    filtered_annotations = []
    small_box_count = 0
    missing_imginfo_count = 0

    for ann in annotations_after_drop:
        img_id = ann.get("image_id")
        if img_id is None:
            continue

        img_info = images_by_id.get(img_id)
        if img_info is None:
            missing_imginfo_count += 1
            continue

        img_w = img_info.get("width")
        img_h = img_info.get("height")
        if not img_w or not img_h:
            missing_imginfo_count += 1
            continue

        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        _, _, bw, bh = bbox
        bbox_area = float(bw) * float(bh)
        img_area = float(img_w) * float(img_h)

        if bbox_area < img_area * MIN_FRACTION_OF_IMAGE:
            small_box_count += 1
            continue

        filtered_annotations.append(ann)

    # -------- 4) Remove images with too many remaining boxes --------
    ann_count_by_image = defaultdict(int)
    for ann in filtered_annotations:
        ann_count_by_image[ann["image_id"]] += 1

    images_too_many = {
        img_id for img_id, count in ann_count_by_image.items()
        if count > MAX_BBOX_PER_IMAGE
    }

    final_annotations = [
        ann for ann in filtered_annotations
        if ann["image_id"] not in images_too_many
    ]

    # -------- 5) Remove images with zero remaining annotations (implicit) --------
    remaining_image_ids = {ann["image_id"] for ann in final_annotations}

    final_images = [
        img for img in images_after_drop
        if img["id"] in remaining_image_ids and img["id"] not in images_too_many
    ]

    stats = {
        "crowd_images_removed": len(crowd_image_ids),
        "excluded_images_removed": len(excluded_image_ids),
        "total_images_removed_by_whole_image_rules": len(drop_image_ids),
        "too_small_annotations_removed": small_box_count,
        "missing_imageinfo_annotations_skipped": missing_imginfo_count,
        "images_removed_too_many_boxes": len(images_too_many),
    }

    return {
        "images": final_images,
        "annotations": final_annotations,
        "categories": categories,  # unchanged
        "_filter_stats": stats,    # optional: easy debugging; remove if you don't want it
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filter COCO instances JSON with whole-image removal for crowd + excluded categories, "
                    "plus bbox size filtering and max-bboxes-per-image."
    )
    parser.add_argument("input_json", type=Path, help="Path to original instances_*.json")
    parser.add_argument("output_json", type=Path, help="Path to write filtered JSON")
    args = parser.parse_args()

    print(f"[info] Loading: {args.input_json}")
    data = load_json(args.input_json)

    n_img_before = len(data.get("images", []))
    n_ann_before = len(data.get("annotations", []))

    print("[info] Filtering...")
    filtered = filter_coco(data)

    # If you don't want this field in your output file, delete it before saving:
    stats = filtered.get("_filter_stats", {})
    # del filtered["_filter_stats"]

    n_img_after = len(filtered.get("images", []))
    n_ann_after = len(filtered.get("annotations", []))

    print(f"[stats] images:      {n_img_before} -> {n_img_after}")
    print(f"[stats] annotations: {n_ann_before} -> {n_ann_after}")
    if stats:
        print(f"[stats] crowd-images removed (had any iscrowd=1): {stats.get('crowd_images_removed', 0)}")
        print(f"[stats] excluded-category images removed (had any excluded category): {stats.get('excluded_images_removed', 0)}")
        print(f"[stats] images removed for whole-image rules (union): {stats.get('total_images_removed_by_whole_image_rules', 0)}")
        print(f"[stats] too-small annotations removed: {stats.get('too_small_annotations_removed', 0)}")
        print(f"[stats] images removed (> MAX_BBOX_PER_IMAGE after filtering): {stats.get('images_removed_too_many_boxes', 0)}")

    print(f"[info] Saving filtered JSON to: {args.output_json}")
    save_json(filtered, args.output_json)
    print("[done] Finished.")


if __name__ == "__main__":
    main()
