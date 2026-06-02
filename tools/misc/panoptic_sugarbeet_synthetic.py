#!/usr/bin/env python3
"""Convert SugarbeetSynthetic annotations to COCO panoptic format.

Input layout (defaults):
- images: <dataset_root>/main_camera/rect/{train,val}/*.png
- instance npz: <dataset_root>/main_camera_annotations/instance_segmentation/<id>.npz
- semantic npz: <dataset_root>/main_camera_annotations/semantic_segmentation/<id>.npz

Output per split:
- <output_root>/<split>/plants_panoptic_<split>.json
- <output_root>/<split>/plants_panoptic_<split>/images/<id>.png

The class ontology is fixed to:
- crop (thing) -> category_id 1
- weed (thing) -> category_id 2
- soil (stuff) -> category_id 3
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


CATEGORIES = [
    {
        'id': 1,
        'name': 'crop',
        'color': [111, 74, 0],
        'supercategory': 'crop',
        'isthing': 1
    },
    {
        'id': 2,
        'name': 'weed',
        'color': [230, 150, 140],
        'supercategory': 'weed',
        'isthing': 1
    },
    {
        'id': 3,
        'name': 'soil',
        'color': [0, 0, 0],
        'supercategory': 'soil',
        'isthing': 0
    },
]


def parse_semantic_to_category(tokens):
    mapping = {}
    for token in tokens:
        if ':' not in token:
            raise ValueError(
                f'Invalid mapping token {token}. Expected semantic_id:category_id')
        s, c = token.split(':', 1)
        mapping[int(s)] = int(c)
    return mapping


def id_to_rgb(segment_id):
    return [
        int(segment_id % 256),
        int((segment_id // 256) % 256),
        int((segment_id // 256 // 256) % 256),
    ]


def compute_bbox(mask):
    ys, xs = np.nonzero(mask)
    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())
    return [x1, y1, x2 - x1 + 1, y2 - y1 + 1]


def load_npz_array(path):
    arr = np.load(path)['array']
    if arr.ndim != 2:
        raise RuntimeError(f'Expected 2D array in {path}, got shape={arr.shape}')
    return arr


def convert_split(split, images_root, instance_root, semantic_root, output_root,
                  sem_to_cat):
    split_img_dir = images_root / split
    if not split_img_dir.is_dir():
        raise FileNotFoundError(f'Missing image split dir: {split_img_dir}')

    image_files = sorted([p for p in split_img_dir.glob('*.png')])
    if not image_files:
        raise RuntimeError(f'No .png images found in {split_img_dir}')

    out_split_dir = output_root / split
    out_pan_dir = out_split_dir / f'plants_panoptic_{split}' / 'images'
    out_json = out_split_dir / f'plants_panoptic_{split}.json'
    out_pan_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []

    missing_instance = 0
    missing_semantic = 0

    for idx, img_path in enumerate(image_files):
        image_id = img_path.stem
        inst_path = instance_root / f'{image_id}.npz'
        sem_path = semantic_root / f'{image_id}.npz'

        if not inst_path.is_file():
            missing_instance += 1
            continue
        if not sem_path.is_file():
            missing_semantic += 1
            continue

        instance_map = load_npz_array(inst_path)
        semantic_map = load_npz_array(sem_path)

        if instance_map.shape != semantic_map.shape:
            raise RuntimeError(
                f'Shape mismatch for {image_id}: '
                f'instance={instance_map.shape}, semantic={semantic_map.shape}')

        h, w = semantic_map.shape
        images.append({
            'id': image_id,
            'width': int(w),
            'height': int(h),
            'file_name': f'{image_id}.png'
        })

        pan_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        segments_info = []
        next_panoptic_id = 1

        # Build segments category-wise. Stuff gets one merged segment.
        for semantic_id, category_id in sem_to_cat.items():
            class_mask = semantic_map == semantic_id
            if not np.any(class_mask):
                continue

            is_thing = category_id in (1, 2)

            if not is_thing:
                segment_masks = [class_mask]
            else:
                instance_ids = np.unique(instance_map[class_mask])
                segment_masks = []
                for inst_id in instance_ids:
                    m = class_mask & (instance_map == inst_id)
                    if np.any(m):
                        segment_masks.append(m)

            for m in segment_masks:
                area = int(m.sum())
                if area <= 0:
                    continue

                seg_id = next_panoptic_id
                next_panoptic_id += 1

                pan_rgb[m] = np.array(id_to_rgb(seg_id), dtype=np.uint8)
                segments_info.append({
                    'id': int(seg_id),
                    'category_id': int(category_id),
                    'area': int(area),
                    'bbox': compute_bbox(m),
                    'iscrowd': 0
                })

        out_mask = out_pan_dir / f'{image_id}.png'
        Image.fromarray(pan_rgb).save(out_mask)

        annotations.append({
            'image_id': image_id,
            'file_name': f'{image_id}.png',
            'segments_info': segments_info
        })

        if (idx + 1) % 200 == 0 or (idx + 1) == len(image_files):
            print(f'[{split}] processed {idx + 1}/{len(image_files)}')

    out_data = {
        'images': images,
        'annotations': annotations,
        'categories': CATEGORIES,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2)

    print(f'[{split}] wrote JSON: {out_json}')
    print(f'[{split}] wrote masks: {out_pan_dir}')
    print(
        f'[{split}] images={len(images)} annotations={len(annotations)} '
        f'missing_instance={missing_instance} missing_semantic={missing_semantic}')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-root',
        required=True,
        help='Root path, e.g. /netscratch/naeem/sugarbeet_syn_v6')
    parser.add_argument(
        '--output-root',
        default=None,
        help='Output root. Defaults to --dataset-root')
    parser.add_argument(
        '--images-root',
        default='main_camera/rect',
        help='Relative path from dataset root to images root with train/val dirs')
    parser.add_argument(
        '--instance-root',
        default='main_camera_annotations/instance_segmentation',
        help='Relative path from dataset root to instance npz files')
    parser.add_argument(
        '--semantic-root',
        default='main_camera_annotations/semantic_segmentation',
        help='Relative path from dataset root to semantic npz files')
    parser.add_argument(
        '--splits', nargs='+', default=['train'])
    parser.add_argument(
        '--semantic-to-category',
        nargs='+',
        default=['0:3', '1:1', '2:2'],
        help='Mapping semantic_id:category_id. Default assumes 0=soil,1=crop,2=weed')

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root) if args.output_root else dataset_root
    images_root = dataset_root / args.images_root
    instance_root = dataset_root / args.instance_root
    semantic_root = dataset_root / args.semantic_root

    sem_to_cat = parse_semantic_to_category(args.semantic_to_category)

    if not instance_root.is_dir():
        raise FileNotFoundError(f'Missing instance-root: {instance_root}')
    if not semantic_root.is_dir():
        raise FileNotFoundError(f'Missing semantic-root: {semantic_root}')

    for split in args.splits:
        convert_split(
            split=split,
            images_root=images_root,
            instance_root=instance_root,
            semantic_root=semantic_root,
            output_root=output_root,
            sem_to_cat=sem_to_cat)


if __name__ == '__main__':
    main()
