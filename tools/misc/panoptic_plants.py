#!/usr/bin/python
#
# Converts the *instanceIds.png annotations of the Cityscapes dataset
# to COCO-style panoptic segmentation format (http://cocodataset.org/#format-data).
# The convertion is working for 'fine' set of the annotations.
#
# By default with this tool uses IDs specified in labels.py. You can use flag
# --use-train-id to get train ids for categories. 'ignoreInEval' categories are
# removed during the conversion.
#
# In panoptic segmentation format image_id is used to match predictions and ground truth.
# For plants image_id has form <city>_123456_123456 and corresponds to the prefix
# of plants image files.
#

# python imports
from __future__ import print_function, absolute_import, division, unicode_literals
import os
import glob
import sys
import argparse
import json
import numpy as np

# Image processing
from PIL import Image

# COCO-panoptic safe category ids for this project.
# Keep segment id 0 reserved for VOID, so we also avoid category id 0.
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

# Merge partial labels into main classes and map semantics to category ids.
# 0=soil, 1=crop, 2=weed, 3=partial_crop, 4=partial_weed
SEMANTIC_TO_CATEGORY = {
    0: 3,
    1: 1,
    2: 2,
    3: 1,
    4: 2,
}



# The main method
def convert2panoptic(plantsPath=None, outputFolder=None, useTrainId=False, setNames=["val", "train", "test"]):
    # Where to look for Cityscapes
    if plantsPath is None:
        if 'DATASET' in os.environ:
            plantsPath = os.environ['DATASET']
        else:
            plantsPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')
        # plantsPath = os.path.join(plantsPath, "gtFine")

    if outputFolder is None:
        outputFolder = plantsPath

    categories = CATEGORIES

    for setName in setNames:
        # how to search for all ground truth
        # searchFine   = os.path.join(plantsPath, setName, "*", "*_instanceIds.png")
        # # search files
        # filesFine = glob.glob(searchFine)
        images_dir = os.path.join(plantsPath, setName, "plant_instances")
        filesFine = os.listdir(images_dir)
        filesFine = [os.path.join(images_dir, file) for file in filesFine]
        filesFine.sort()

        files = filesFine
        # quit if we did not find anything
        if not files:
            print(
                "Did not find any files for {} set using matching pattern {}. Please consult the README.".format(setName, "ciao")
            )
        # a bit verbose
        print("Converting {} annotation files for {} set.".format(len(files), setName))

        trainIfSuffix = "" #"_trainId" if useTrainId else ""
        outputBaseFile = "plants_panoptic_{}{}".format(setName, trainIfSuffix)
        outFile = os.path.join(outputFolder, "{}.json".format(outputBaseFile))
        print("Json file with the annotations in panoptic format will be saved in {}".format(outFile))
        panopticFolder = os.path.join(outputFolder, outputBaseFile)
        if not os.path.isdir(panopticFolder):
            print("Creating folder {} for panoptic segmentation PNGs".format(panopticFolder))
            os.mkdir(panopticFolder)
        print("Corresponding segmentations in .png format will be saved in {}".format(panopticFolder))

        images = []
        annotations = []
        for progress, f in enumerate(files):

            originalFormat = np.array(Image.open(f))
            semantic_path = f.replace("plant_instances", "semantics")
            semantics = np.array(Image.open(semantic_path))

            fileName = os.path.basename(f)
            imageId = fileName.replace(".png", "")
            inputFileName = os.path.join("images", fileName)
            outputFileName = fileName
            # image entry, id for image is its filename without extension
            images.append({"id": imageId,
                           "width": int(originalFormat.shape[1]),
                           "height": int(originalFormat.shape[0]),
                           "file_name": inputFileName})

            pan_format = np.zeros(
                (originalFormat.shape[0], originalFormat.shape[1], 3), dtype=np.uint8
            )

            segmentIds = np.unique(originalFormat)
            segmInfo = []
            next_panoptic_id = 1
            for segmentId in segmentIds:
                mask = originalFormat == segmentId
                semanticId = int(semantics[mask][0])
                categoryId = SEMANTIC_TO_CATEGORY.get(semanticId)
                if categoryId is None:
                    continue

                panopticId = next_panoptic_id
                next_panoptic_id += 1

                color = [panopticId % 256, panopticId // 256,
                         panopticId // 256 // 256]
                pan_format[mask] = color

                area = np.sum(mask) # segment area computation

                # bbox computation for a segment
                hor = np.sum(mask, axis=0)
                hor_idx = np.nonzero(hor)[0]
                x = hor_idx[0]
                width = hor_idx[-1] - x + 1
                vert = np.sum(mask, axis=1)
                vert_idx = np.nonzero(vert)[0]
                y = vert_idx[0]
                height = vert_idx[-1] - y + 1
                bbox = [int(x), int(y), int(width), int(height)]

                segmInfo.append({"id": int(panopticId),
                                 "category_id": int(categoryId),
                                 "area": int(area),
                                 "bbox": bbox,
                                 "iscrowd": 0})

            annotations.append({'image_id': imageId,
                                'file_name': outputFileName,
                                "segments_info": segmInfo})

            Image.fromarray(pan_format).save(os.path.join(panopticFolder, outputFileName))

            print("\rProgress: {:>3.2f} %".format((progress + 1) * 100 / len(files)), end=' ')
            sys.stdout.flush()

        print("\nSaving the json file {}".format(outFile))
        d = {'images': images,
             'annotations': annotations,
             'categories': categories}
        with open(outFile, 'w') as f:
            json.dump(d, f, sort_keys=True, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-folder",
                        dest="plantsPath",
                        help="path to the Cityscapes dataset 'gtFine' folder",
                        default=None,
                        type=str)
    parser.add_argument("--output-folder",
                        dest="outputFolder",
                        help="path to the output folder.",
                        default=None,
                        type=str)
    parser.add_argument("--use-train-id", action="store_true", dest="useTrainId")
    parser.add_argument("--set-names",
                        dest="setNames",
                        help="set names to which apply the function to",
                        nargs='+',
                        default=["val", "train", "test"],
                        type=str)
    args = parser.parse_args()

    convert2panoptic(args.plantsPath, args.outputFolder, args.useTrainId, args.setNames)


# call the main
if __name__ == "__main__":
    main()