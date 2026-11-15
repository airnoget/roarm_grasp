"""Reviewed single-class boxes, native RGB coordinates, and YOLO label serialization."""
import json
import math
import os
from pathlib import Path
import time

CLASS_NAME = 'artificial_object'


def atomic_text(path, text):
    path = Path(path)
    temp = path.with_name(path.name+f'.{time.time_ns()}.tmp')
    temp.write_text(text, encoding='utf-8')
    os.replace(temp, path)


def normalize_boxes(boxes, width, height):
    if width <= 0 or height <= 0:
        raise ValueError('Invalid image size')
    labels = []
    for box in boxes:
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise ValueError('Invalid box coordinates')
        x0, y0, x1, y1 = map(float, box)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height) or x1-x0 < 2 or y1-y0 < 2:
            raise ValueError('Box must lie inside image and be at least 2 pixels wide/high')
        labels.append([0, (x0+x1)/2/width, (y0+y1)/2/height, (x1-x0)/width, (y1-y0)/height])
    return labels


def save_labels(folder, boxes, image_size, confirmed_empty=False):
    folder = Path(folder)
    meta = json.loads((folder/'capture.json').read_text(encoding='utf-8-sig'))
    image = meta['annotation_image']
    if (folder/image).resolve().parent != folder.resolve() or not (folder/image).is_file():
        raise ValueError('Annotation image must exist inside capture directory')
    width, height = image_size
    labels = normalize_boxes(boxes, width, height)
    if not labels and not confirmed_empty:
        raise ValueError('Draw all targets, or explicitly confirm no target using B')
    if labels and confirmed_empty:
        raise ValueError('Cannot mark a frame with boxes as empty')
    annotation = dict(schema=1, image=image, image_size=[width, height],
                      class_names={0: CLASS_NAME}, boxes_xyxy=[list(map(float, b)) for b in boxes],
                      confirmed_empty=bool(confirmed_empty), reviewed_by_user=True,
                      reviewed_time_ns=time.time_ns(), source='manual_boxes')
    label_name = Path(image).with_suffix('.txt').name
    text = ''.join(' '.join([str(row[0]), *(f'{v:.8f}' for v in row[1:])])+'\n' for row in labels)
    atomic_text(folder/'annotation.json', json.dumps(annotation, indent=2))
    atomic_text(folder/label_name, text)
    meta.update(annotation_status='reviewed', contains_target=bool(labels),
                annotation_file='annotation.json', yolo_label_file=label_name,
                class_names={0: CLASS_NAME})
    atomic_text(folder/'capture.json', json.dumps(meta, indent=2))
    return annotation
