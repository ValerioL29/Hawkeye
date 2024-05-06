import json
from collections import defaultdict

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from .utils import make_dirs, coco91_to_coco80_class

import cv2
from pycocotools import mask as cocomask


def rle_to_coco(annotation: dict) -> dict:
    """Transform the rle coco annotation (a single one) into coco style.
    In this case, one mask can contain several polygons, later leading to several `Annotation` objects.
    In case of not having a valid polygon (the mask is a single pixel) it will be an empty list.

    Refer to: https://stackoverflow.com/questions/75326066/coco-annotations-convert-rle-to-polygon-segmentation

    Parameters
    ----------
    annotation : dict
        rle coco style annotation
    Returns
    -------
    list[dict]
        list of coco style annotations (in dict format)
    """
    tmp_ann = annotation.copy()
    masked_arr = cocomask.decode(annotation["segmentation"])
    contours, _ = cv2.findContours(
        masked_arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    segmentations = [contour for contour in contours if contour.size >= 6]
    seg_coords = [
        seg.astype(float).flatten().tolist() for seg in segmentations
    ]
    # Replace the original segmentation object with the new one
    tmp_ann["segmentation"] = seg_coords

    return tmp_ann


def min_index(arr1, arr2):
    """
    Find a pair of indexes with the shortest distance.

    Args:
        arr1: (N, 2).
        arr2: (M, 2).
    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """
    Merge multi segments to one list. Find the coordinates with min distance between each segment, then connect these
    coordinates with one thin line to merge all segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...],
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def delete_dsstore(path="../datasets"):
    """Deletes Apple .DS_Store files recursively from a specified directory."""
    from pathlib import Path

    files = list(Path(path).rglob(".DS_store"))
    print(files)
    for f in files:
        f.unlink()


def convert_coco_json(
    json_dir="../coco/annotations/",
    save_dir: Path = make_dirs(),  # output directory
    use_segments=False,
    cls91to80=False,
    use_rle=False,
):
    """Converts COCO JSON format to YOLO label format, with options for segments and class mapping."""
    # save_dir = make_dirs()  # output directory
    _ = os.makedirs(save_dir, exist_ok=True)
    coco80 = coco91_to_coco80_class()

    # Import json
    for json_file in sorted(Path(json_dir).resolve().glob("*.json")):
        fn = save_dir
        with open(json_file, "r+") as f:
            data = json.load(f)

        # Create image dict
        images = {"%g" % x["id"]: x for x in data["images"]}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        # Write labels file
        for img_id, anns in tqdm(
            imgToAnns.items(), desc=f"Annotations {json_file}"
        ):
            img = images["%g" % img_id]
            h, w, f = img["height"], img["width"], img["file_name"]

            bboxes = []
            segments = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = (
                    coco80[ann["category_id"] - 1]
                    if cls91to80
                    else ann["category_id"] - 1
                )  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                # Segments
                if use_segments:
                    if len(ann["segmentation"]) > 1:
                        s = merge_multi_segment(ann["segmentation"])
                        s = (
                            (np.concatenate(s, axis=0) / np.array([w, h]))
                            .reshape(-1)
                            .tolist()
                        )
                    else:
                        # s = [
                        #     j for i in ann["segmentation"] for j in i
                        # ]  # all segments concatenated
                        s = (
                            (np.array(s).reshape(-1, 2) / np.array([w, h]))
                            .reshape(-1)
                            .tolist()
                        )
                    s = [cls] + s
                    if s not in segments:
                        segments.append(s)

            # Write
            with open((fn / f).with_suffix(".txt"), "a") as file:
                for i in range(len(bboxes)):
                    line = (
                        *(segments[i] if use_segments else bboxes[i]),
                    )  # cls, box or segments
                    file.write(("%g " * len(line)).rstrip() % line + "\n")
