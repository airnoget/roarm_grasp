"""Validate reviewed RGB boxes and export a reproducible, grouped YOLO split."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'.deps'))
from PIL import Image
import numpy as np
from dataset_labels import CLASS_NAME, atomic_text, normalize_boxes


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def child(folder, name):
    path = (folder/name).resolve()
    if path.parent != folder.resolve() or not path.is_file():
        raise ValueError(f'Missing or unsafe source file: {path}')
    return path


def records(session):
    result = []
    for manifest in sorted(Path(session).glob('*/capture.json')):
        folder = manifest.parent
        meta = read_json(manifest)
        profile=meta.get('camera_profile')
        if profile is not None:
            snapshot=(folder.parent/profile['preset_file']).resolve()
            if snapshot.parent!=folder.parent.resolve() or not snapshot.is_file():
                raise ValueError(f'Missing or unsafe preset snapshot: {folder.name}')
            if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=profile.get('preset_sha256'):
                raise ValueError(f'Preset snapshot changed: {folder.name}')
        if meta.get('annotation_status') != 'reviewed':
            raise ValueError(f'Unreviewed capture: {folder.name}; finish python debug.py label')
        image = child(folder, meta['annotation_image'])
        annotation = read_json(child(folder, meta['annotation_file']))
        if annotation.get('reviewed_by_user') is not True or annotation['image'] != image.name:
            raise ValueError(f'Invalid annotation identity: {folder.name}')
        if annotation.get('class_names') != {'0': CLASS_NAME}:
            raise ValueError(f'Unexpected label classes: {folder.name}')
        with Image.open(image) as im:
            size = list(im.size)
            im.verify()
        if annotation['image_size'] != size:
            raise ValueError(f'Image size changed: {folder.name}')
        labels = normalize_boxes(annotation['boxes_xyxy'], *size)
        if not labels and annotation.get('confirmed_empty') is not True:
            raise ValueError(f'Empty frame not confirmed: {folder.name}')
        if labels and annotation.get('confirmed_empty'):
            raise ValueError(f'Contradictory empty annotation: {folder.name}')
        if meta.get('contains_target') is not bool(labels):
            raise ValueError(f'Target presence mismatch: {folder.name}')
        text = child(folder, meta['yolo_label_file']).read_text(encoding='utf-8')
        parsed = [[float(v) for v in line.split()] for line in text.splitlines() if line.strip()]
        if np.asarray(parsed).shape != np.asarray(labels).shape or not np.allclose(parsed, labels, atol=1e-7, rtol=0):
            raise ValueError(f'YOLO label disagrees with manual boxes: {folder.name}')
        result.append(dict(id=folder.name, image=image, labels=labels,
                           group=str(meta.get('split_group') or meta.get('scene_id') or folder.name),
                           stage=meta.get('collection_stage', 'unknown'),
                           camera_profile=meta.get('camera_profile'),
                           sha256=hashlib.sha256(image.read_bytes()).hexdigest()))
    if not result:
        raise ValueError('No complete captures found')
    return result


def split_records(rows, seed=42):
    # Union scene groups and byte-identical images: neither may cross the split.
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for key in [('scene', row['group']), ('image', row['sha256'])]:
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    strata = defaultdict(list)
    for group in groups.values():
        strata[tuple(sorted(set(r['stage'] for r in group)))].append(group)
    rng = random.Random(seed)
    splits = dict(train=[], val=[])
    for stage in sorted(strata):
        groups = strata[stage]
        if len(groups) < 2:
            raise ValueError(f'Need at least two independent scenes for stage {stage}')
        rng.shuffle(groups)
        n = max(1, min(len(groups)-1, round(len(groups)*.2)))
        for i, group in enumerate(groups):
            splits['val' if i < n else 'train'].extend(group)
    for name, split in splits.items():
        if not any(row['labels'] for row in split):
            raise ValueError(f'{name} contains no positive examples')
    return splits


def export(session, destination, seed=42):
    rows = records(session)
    splits = split_records(rows, seed)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    hashes={row['camera_profile'].get('preset_sha256') for row in rows if row['camera_profile']}
    if hashes and (len(hashes)!=1 or any(not row['camera_profile'] for row in rows)):
        raise ValueError('Do not mix unknown or different camera presets in one export')
    manifest = dict(schema=1, source_session=str(Path(session).resolve()), seed=seed,
                    camera_preset_sha256=next(iter(hashes)) if hashes else None,
                    class_names={0: CLASS_NAME}, split_policy='scene-and-exact-image-grouped; 80/20 per capture stage',
                    scope='small baseline validation, not an independent competition test', splits={})
    for name, split in splits.items():
        image_dir = destination/'images'/name
        label_dir = destination/'labels'/name
        image_dir.mkdir(parents=True); label_dir.mkdir(parents=True)
        manifest['splits'][name] = []
        for row in split:
            shutil.copy2(row['image'], image_dir/(row['id']+'.png'))
            text = ''.join('0 '+' '.join(f'{x:.8f}' for x in label[1:])+'\n' for label in row['labels'])
            (label_dir/(row['id']+'.txt')).write_text(text, encoding='utf-8')
            manifest['splits'][name].append({k: v for k, v in row.items() if k not in ('image', 'labels')} |
                                            dict(source_image=str(row['image'].resolve()), boxes=len(row['labels'])))
    yaml = 'path: '+json.dumps(destination.as_posix())+'\ntrain: images/train\nval: images/val\nnames:\n  0: '+CLASS_NAME+'\n'
    (destination/'dataset.yaml').write_text(yaml, encoding='utf-8')
    atomic_text(destination/'manifest.json', json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.session is None:
        sessions = [p for p in (ROOT/'datasets/competition/raw').glob('session-*') if any(p.glob('*/capture.json'))]
        if not sessions: parser.error('No captures found')
        args.session = max(sessions, key=lambda p: p.name)
    output = args.output or ROOT/'datasets/competition/yolo'/f'export-{time.time_ns()}'
    try:
        manifest = export(args.session, output)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f'EXPORT FAILED: {exc}\n')
    atomic_text(ROOT/'datasets/competition/latest-export.json', json.dumps(dict(dataset=str((output/'dataset.yaml').resolve())), indent=2))
    for name, rows in manifest['splits'].items():
        print(name, 'images=',len(rows), 'boxes=',sum(r['boxes'] for r in rows), 'backgrounds=',sum(r['boxes']==0 for r in rows), 'stages=',dict(Counter(r['stage'] for r in rows)))
    print('DATASET', (output/'dataset.yaml').resolve())
    print('NEXT: python debug.py train')


if __name__ == '__main__': main()
