#!/usr/bin/env python3
"""Validate a COCO panoptic-style dataset on disk.

This script is intended as a proof-check utility before training.
It validates:
1) JSON schema-level consistency (images/annotations/categories)
2) Category rules (expected class names and thing/stuff flags)
3) Per-mask consistency between PNG segment ids and JSON segments_info
4) Optional RGB image existence checks
"""

import argparse
import json
import os
import re
from collections import Counter

import numpy as np
from PIL import Image


def rgb2id(color):
    """Convert RGB panoptic encoding to integer ids.

    COCO panopticapi definition:
    id = R + 256 * G + 256 * 256 * B
    """
    if color.ndim == 2:
        return color.astype(np.int64)
    return color[:, :, 0].astype(np.int64) + 256 * color[:, :, 1].astype(
        np.int64) + 256 * 256 * color[:, :, 2].astype(np.int64)


def add_error(store, key, msg):
    store[key].append(msg)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate COCO panoptic annotations and masks.')
    parser.add_argument(
        '--ann-json',
        required=True,
        help='Path to panoptic json file, e.g. plants_panoptic_val.json')
    parser.add_argument(
        '--panoptic-root',
        required=True,
        help='Directory containing panoptic png files referenced by JSON.')
    parser.add_argument(
        '--images-root',
        default=None,
        help='Optional directory containing RGB image files from images[].')
    parser.add_argument(
        '--max-mask-checks',
        type=int,
        default=0,
        help='If >0, only validate first N masks. 0 means validate all.')
    parser.add_argument(
        '--expected-categories',
        nargs='*',
        default=['soil:0', 'crop:1', 'weed:1'],
        help=('Expected category rules in the form name:isthing. '
              'Default enforces background(stuff), crop(thing), weed(thing).'))
    parser.add_argument(
        '--expected-category-ids',
        nargs='*',
          default=['crop:1', 'weed:2', 'soil:3'],
        help=('Expected category ids in the form name:id. '
              'Default enforces crop=1, weed=2, soil=3.'))
    parser.add_argument(
        '--config-file',
        default=None,
        help=('Optional MMDet dataset config path. If provided, the script '
              'verifies that --ann-json and --panoptic-root are consistent '
              'with ann_file/seg_prefix entries in that config.'))
    parser.add_argument(
        '--background-category-name',
        default='soil',
        help='Background/stuff class name to enforce in per-image checks.')
    parser.add_argument(
        '--background-category-id',
        type=int,
        default=None,
        help=('Optional explicit background category id. If omitted, it is '
              'resolved by --background-category-name.'))
    parser.add_argument(
        '--background-min-area-ratio',
        type=float,
          default=0.45,
        help=('Minimum background area ratio within annotated area per image. '
              'Default: 0.45'))
    parser.add_argument(
        '--background-min-bbox-ratio',
        type=float,
        default=0.98,
        help=('Background bbox must span at least this fraction of image '
              'width and height. Default: 0.98'))
    return parser.parse_args()


def _resolve_path(data_root, value):
    if os.path.isabs(value):
        return os.path.normpath(value)
    return os.path.normpath(os.path.join(data_root, value))


def parse_mmdet_config_paths(config_file):
    """Best-effort parser for data_root, ann_file, seg_prefix in config text."""
    with open(config_file, 'r', encoding='utf-8') as f:
        text = f.read()

    data_root_match = re.search(r"^\s*data_root\s*=\s*['\"]([^'\"]+)['\"]", text,
                                flags=re.MULTILINE)
    data_root = data_root_match.group(1) if data_root_match else ''

    ann_matches = re.findall(
        r"ann_file\s*=\s*(?:data_root\s*\+\s*)?['\"]([^'\"]+)['\"]", text)
    seg_matches = re.findall(
        r"seg_prefix\s*=\s*(?:data_root\s*\+\s*)?['\"]([^'\"]+)['\"]", text)

    ann_paths = {_resolve_path(data_root, p) for p in ann_matches}
    seg_paths = {_resolve_path(data_root, p) for p in seg_matches}
    return data_root, ann_paths, seg_paths


def main():
    args = parse_args()

    with open(args.ann_json, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    errors = Counter()
    details = {
        'config': [],
        'json': [],
        'categories': [],
        'images': [],
        'annotations': [],
        'files': [],
        'segments': [],
    }

    for key in ('images', 'annotations', 'categories'):
        if key not in coco:
            add_error(details, 'json', f'Missing top-level key: {key}')
            errors['json'] += 1

    if errors['json'] > 0:
        print('FAILED: annotation JSON is missing required top-level keys.')
        for msg in details['json']:
            print(f'  - {msg}')
        raise SystemExit(2)

    # Optional config consistency checks.
    if args.config_file is not None:
        if not os.path.isfile(args.config_file):
            add_error(details, 'config', f'Config file not found: {args.config_file}')
            errors['config'] += 1
        else:
            data_root, ann_paths, seg_paths = parse_mmdet_config_paths(
                args.config_file)
            ann_json_abs = os.path.normpath(os.path.abspath(args.ann_json))
            panoptic_root_abs = os.path.normpath(os.path.abspath(args.panoptic_root))

            if ann_json_abs not in ann_paths:
                add_error(
                    details,
                    'config',
                    'Provided --ann-json does not match any ann_file in config. '
                    f'Given: {ann_json_abs}; config ann_file paths: {sorted(ann_paths)}')
                errors['config'] += 1

            if panoptic_root_abs not in seg_paths:
                add_error(
                    details,
                    'config',
                    'Provided --panoptic-root does not match any seg_prefix in config. '
                    f'Given: {panoptic_root_abs}; config seg_prefix paths: {sorted(seg_paths)}')
                errors['config'] += 1

            if panoptic_root_abs.endswith('/images') or panoptic_root_abs.endswith('\\images'):
                # Some datasets store panoptic masks under a nested folder named
                # "images". Only flag this when annotation file names do not
                # resolve under the provided root.
                missing_probe = 0
                if 'annotations' in coco and isinstance(coco['annotations'], list):
                    for ann in coco['annotations'][:50]:
                        f = ann.get('file_name')
                        if not f:
                            continue
                        if not os.path.isfile(os.path.join(panoptic_root_abs, f)):
                            missing_probe += 1
                if missing_probe > 0:
                    add_error(
                        details,
                        'config',
                        'Panoptic root ends with "images" and some annotation masks '
                        'do not resolve there. Verify seg_prefix points to panoptic '
                        f'masks. Missing in probe: {missing_probe}/50')
                    errors['config'] += 1

    images = coco['images']
    annotations = coco['annotations']
    categories = coco['categories']

    # Category checks
    category_by_id = {}
    category_by_name = {}
    for c in categories:
        cid = c.get('id')
        name = c.get('name')
        isthing = c.get('isthing')
        if cid in category_by_id:
            add_error(details, 'categories', f'Duplicate category id: {cid}')
            errors['categories'] += 1
        category_by_id[cid] = c

        if name in category_by_name:
            add_error(details, 'categories', f'Duplicate category name: {name}')
            errors['categories'] += 1
        category_by_name[name] = c

        if isthing not in (0, 1):
            add_error(
                details,
                'categories',
                f'Category {name} (id={cid}) has invalid isthing={isthing}')
            errors['categories'] += 1
        if isinstance(cid, int) and cid == 0:
            add_error(
                details,
                'categories',
                'Category id 0 detected. In COCO panoptic, segment id 0 is VOID; '
                'use positive category ids to avoid ambiguity.')
            errors['categories'] += 1

    expected = {}
    for token in args.expected_categories:
        if ':' not in token:
            print(
                f"Invalid --expected-categories token '{token}', expected name:isthing"
            )
            raise SystemExit(2)
        n, t = token.split(':', 1)
        expected[n] = int(t)

    expected_ids = {}
    for token in args.expected_category_ids:
        if ':' not in token:
            print(
                f"Invalid --expected-category-ids token '{token}', expected name:id"
            )
            raise SystemExit(2)
        n, cid = token.split(':', 1)
        expected_ids[n] = int(cid)

    if len(categories) != len(expected):
        add_error(
            details,
            'categories',
            f'Expected {len(expected)} categories but JSON has {len(categories)}')
        errors['categories'] += 1

    for name, isthing in expected.items():
        if name not in category_by_name:
            add_error(details, 'categories', f'Missing expected category: {name}')
            errors['categories'] += 1
            continue
        got = category_by_name[name].get('isthing')
        if got != isthing:
            add_error(
                details,
                'categories',
                f'Category {name} expected isthing={isthing}, got {got}')
            errors['categories'] += 1

    for name, expected_id in expected_ids.items():
        if name not in category_by_name:
            add_error(details, 'categories', f'Missing expected category for id check: {name}')
            errors['categories'] += 1
            continue
        got_id = category_by_name[name].get('id')
        if got_id != expected_id:
            add_error(
                details,
                'categories',
                f'Category {name} expected id={expected_id}, got {got_id}')
            errors['categories'] += 1

    extra_names = set(category_by_name.keys()) - set(expected.keys())
    for name in sorted(extra_names):
        add_error(details, 'categories', f'Unexpected extra category: {name}')
        errors['categories'] += 1

    if args.background_category_id is not None:
        bg_cat_id = args.background_category_id
    else:
        bg_cat = category_by_name.get(args.background_category_name)
        bg_cat_id = bg_cat.get('id') if bg_cat is not None else None

    if bg_cat_id is None:
        add_error(
            details,
            'categories',
            f'Background category not found by name: {args.background_category_name}')
        errors['categories'] += 1

    # Image table checks
    image_by_id = {}
    image_file_names = set()
    for im in images:
        iid = im.get('id')
        file_name = im.get('file_name')
        width = im.get('width')
        height = im.get('height')
        if iid in image_by_id:
            add_error(details, 'images', f'Duplicate image id: {iid}')
            errors['images'] += 1
        image_by_id[iid] = im

        if file_name in image_file_names:
            add_error(details, 'images', f'Duplicate image file_name: {file_name}')
            errors['images'] += 1
        image_file_names.add(file_name)

        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            add_error(
                details,
                'images',
                f'Image id={iid} has invalid width/height: {width}x{height}')
            errors['images'] += 1

        if args.images_root is not None:
            img_path = os.path.join(args.images_root, file_name)
            if not os.path.isfile(img_path):
                add_error(details, 'files', f'Missing RGB image file: {img_path}')
                errors['files'] += 1

    # Annotation table checks
    ann_by_image = {}
    bg_area_ratios = []
    for ann in annotations:
        image_id = ann.get('image_id')
        seg_file = ann.get('file_name')
        segments_info = ann.get('segments_info')

        if image_id not in image_by_id:
            add_error(
                details,
                'annotations',
                f'Annotation references unknown image_id: {image_id}')
            errors['annotations'] += 1

        if image_id in ann_by_image:
            add_error(
                details,
                'annotations',
                f'Multiple panoptic annotations for image_id={image_id}')
            errors['annotations'] += 1
        ann_by_image[image_id] = ann

        if not isinstance(segments_info, list):
            add_error(
                details,
                'annotations',
                f'image_id={image_id} has non-list segments_info')
            errors['annotations'] += 1
            continue

        image_info = image_by_id.get(image_id, {})
        image_h = image_info.get('height', 0)
        image_w = image_info.get('width', 0)

        seg_ids = set()
        for s in segments_info:
            sid = s.get('id')
            cat_id = s.get('category_id')
            area = s.get('area')
            bbox = s.get('bbox')

            if sid in seg_ids:
                add_error(
                    details,
                    'annotations',
                    f'image_id={image_id} has duplicate segment id {sid}')
                errors['annotations'] += 1
            seg_ids.add(sid)

            if cat_id not in category_by_id:
                add_error(
                    details,
                    'annotations',
                    f'image_id={image_id} segment_id={sid} has unknown category_id={cat_id}'
                )
                errors['annotations'] += 1

            if not isinstance(area, (int, float)) or area <= 0:
                add_error(
                    details,
                    'annotations',
                    f'image_id={image_id} segment_id={sid} has invalid area={area}'
                )
                errors['annotations'] += 1

            if not isinstance(bbox, list) or len(bbox) != 4:
                add_error(
                    details,
                    'annotations',
                    f'image_id={image_id} segment_id={sid} has invalid bbox={bbox}'
                )
                errors['annotations'] += 1

        # Background/stuff quality checks to diagnose stuff-PQ=0.
        if bg_cat_id is not None:
            bg_segments = [s for s in segments_info if s.get('category_id') == bg_cat_id]
            if len(bg_segments) == 0:
                add_error(
                    details,
                    'annotations',
                    f'image_id={image_id} has no background category_id={bg_cat_id} segment')
                errors['annotations'] += 1
            else:
                bg_area = int(sum(int(s.get('area', 0)) for s in bg_segments))
                total_area = int(sum(int(s.get('area', 0)) for s in segments_info))
                max_any_area = int(max(int(s.get('area', 0)) for s in segments_info))

                if total_area > 0:
                    bg_ratio = bg_area / float(total_area)
                    bg_area_ratios.append(bg_ratio)
                    if bg_ratio < args.background_min_area_ratio:
                        add_error(
                            details,
                            'annotations',
                            f'image_id={image_id} background area ratio too low: '
                            f'{bg_ratio:.4f} < {args.background_min_area_ratio}')
                        errors['annotations'] += 1

                if bg_area < max_any_area:
                    add_error(
                        details,
                        'annotations',
                        f'image_id={image_id} background area ({bg_area}) is not '
                        f'the largest segment area ({max_any_area})')
                    errors['annotations'] += 1

                if image_w > 0 and image_h > 0:
                    bg_bbox_ok = False
                    min_w = args.background_min_bbox_ratio * image_w
                    min_h = args.background_min_bbox_ratio * image_h
                    for s in bg_segments:
                        x, y, bw, bh = s.get('bbox', [0, 0, 0, 0])
                        if bw >= min_w and bh >= min_h:
                            bg_bbox_ok = True
                            break
                    if not bg_bbox_ok:
                        add_error(
                            details,
                            'annotations',
                            f'image_id={image_id} background bbox does not span '
                            f'>= {args.background_min_bbox_ratio:.2f} of image size')
                        errors['annotations'] += 1

        mask_path = os.path.join(args.panoptic_root, seg_file)
        if not os.path.isfile(mask_path):
            add_error(details, 'files', f'Missing panoptic mask file: {mask_path}')
            errors['files'] += 1

    # Ensure every image has exactly one annotation.
    for iid in image_by_id:
        if iid not in ann_by_image:
            add_error(details, 'annotations', f'No annotation for image_id={iid}')
            errors['annotations'] += 1

    # Per-mask pixel checks.
    checked = 0
    for image_id, ann in ann_by_image.items():
        if args.max_mask_checks > 0 and checked >= args.max_mask_checks:
            break

        seg_file = ann.get('file_name')
        mask_path = os.path.join(args.panoptic_root, seg_file)
        if not os.path.isfile(mask_path):
            continue

        im_info = image_by_id.get(image_id)
        if im_info is None:
            continue

        try:
            mask = np.asarray(Image.open(mask_path))
        except Exception as exc:
            add_error(
                details,
                'files',
                f'Failed reading panoptic mask {mask_path}: {exc}')
            errors['files'] += 1
            continue

        if mask.ndim not in (2, 3):
            add_error(
                details,
                'segments',
                f'image_id={image_id} has unsupported mask shape={mask.shape}')
            errors['segments'] += 1
            continue

        h, w = mask.shape[:2]
        if h != im_info.get('height') or w != im_info.get('width'):
            add_error(
                details,
                'segments',
                f'image_id={image_id} dimension mismatch: json={im_info.get("width")}x{im_info.get("height")}, '
                f'mask={w}x{h}')
            errors['segments'] += 1

        pan_ids = rgb2id(mask)
        present_ids = set(np.unique(pan_ids).tolist())

        json_ids = set(int(s['id']) for s in ann.get('segments_info', []))

        missing_in_mask = sorted(json_ids - present_ids)
        if missing_in_mask:
            add_error(
                details,
                'segments',
                f'image_id={image_id} segment ids in JSON but absent in mask: {missing_in_mask[:20]}'
            )
            errors['segments'] += len(missing_in_mask)

        extra_in_mask = sorted([sid for sid in (present_ids - json_ids) if sid != 0])
        if extra_in_mask:
            add_error(
                details,
                'segments',
                f'image_id={image_id} segment ids in mask but absent in JSON: {extra_in_mask[:20]}'
            )
            errors['segments'] += len(extra_in_mask)

        # Check area from mask exactly matches JSON area.
        for s in ann.get('segments_info', []):
            sid = int(s['id'])
            expected_area = int(round(float(s['area'])))
            actual_area = int((pan_ids == sid).sum())
            if actual_area != expected_area:
                add_error(
                    details,
                    'segments',
                    f'image_id={image_id} segment_id={sid} area mismatch: json={expected_area}, mask={actual_area}'
                )
                errors['segments'] += 1

        checked += 1

    total_errors = sum(errors.values())
    print('=== Panoptic Dataset Validation Report ===')
    print(f'Annotation JSON: {args.ann_json}')
    print(f'Panoptic root:   {args.panoptic_root}')
    print(f'Images root:     {args.images_root if args.images_root else "(skipped)"}')
    print(f'Images:          {len(images)}')
    print(f'Annotations:     {len(annotations)}')
    print(f'Categories:      {len(categories)}')
    print(f'Masks checked:   {checked}')
    if bg_area_ratios:
        print('Background area ratio '
              f'(min/mean/max): {min(bg_area_ratios):.4f}/'
              f'{sum(bg_area_ratios) / len(bg_area_ratios):.4f}/'
              f'{max(bg_area_ratios):.4f}')
    print('')
    print('Error counts by type:')
    for key in ('config', 'json', 'categories', 'images', 'annotations', 'files', 'segments'):
        print(f'  {key:12s}: {errors[key]}')

    if total_errors == 0:
        print('\nPASS: Dataset looks consistent for COCO panoptic + expected class rules.')
        raise SystemExit(0)

    print('\nFAIL: Found dataset consistency issues.')
    # Print compact details with caps to avoid flooding terminal.
    max_lines_per_group = 20
    for group in ('config', 'categories', 'images', 'annotations', 'files', 'segments'):
        msgs = details[group]
        if not msgs:
            continue
        print(f'\n[{group}] showing {min(len(msgs), max_lines_per_group)} of {len(msgs)}:')
        for msg in msgs[:max_lines_per_group]:
            print(f'  - {msg}')

    raise SystemExit(1)


if __name__ == '__main__':
    main()