#!/usr/bin/env python3
"""Remap category IDs in COCO panoptic JSON without changing mask PNGs.

Useful when category ids were generated in an order that conflicts with
MMDet metainfo class order.
"""

import argparse
import json
import os


def parse_id_map(tokens):
    mapping = {}
    for token in tokens:
        if ':' not in token:
            raise ValueError(f'Invalid --id-map token: {token}. Expected old:new')
        old_s, new_s = token.split(':', 1)
        mapping[int(old_s)] = int(new_s)
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-json', required=True)
    parser.add_argument('--output-json', default=None)
    parser.add_argument(
        '--id-map',
        nargs='+',
        default=['1:3', '2:1', '3:2'],
        help='Category-id remap old:new. Default maps soil/crop/weed ids '
             '1/2/3 -> 3/1/2 for metainfo classes (crop,weed,soil).')
    args = parser.parse_args()

    id_map = parse_id_map(args.id_map)
    output_json = args.output_json if args.output_json else args.input_json

    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'categories' not in data or 'annotations' not in data:
        raise RuntimeError('Input does not look like COCO panoptic JSON.')

    # Remap categories table IDs.
    for cat in data['categories']:
        old_id = int(cat['id'])
        if old_id in id_map:
            cat['id'] = int(id_map[old_id])

    # Remap segment category IDs.
    for ann in data['annotations']:
        for seg in ann.get('segments_info', []):
            old_id = int(seg['category_id'])
            if old_id in id_map:
                seg['category_id'] = int(id_map[old_id])

    # Keep categories sorted by id for readability.
    data['categories'] = sorted(data['categories'], key=lambda x: int(x['id']))

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f'Remapped category IDs written to: {output_json}')
    print(f'id_map used: {id_map}')


if __name__ == '__main__':
    main()
