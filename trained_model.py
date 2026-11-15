"""Resolve the completed local detector training without changing old defaults."""
import hashlib
import json
from pathlib import Path


def latest_detector(root):
    root=Path(root)
    bundled=root/'weights/targets-best.pt'
    pointer = Path(root)/'datasets/competition/latest-training.json'
    if not pointer.is_file():
        if bundled.is_file():return bundled
        raise ValueError('No completed training. Run python debug.py train first.')
    record = json.loads(pointer.read_text(encoding='utf-8-sig'))
    if record.get('class_names') != {'0':'artificial_object'}:
        raise ValueError('Unexpected trained detector classes')
    weights = Path(record['best_weights'])
    if not weights.is_file():
        if bundled.is_file():return bundled
        raise ValueError(f'Missing trained weights: {weights}')
    verification = weights.parent.parent/'verification.json'
    if verification.is_file():
        expected = json.loads(verification.read_text(encoding='utf-8'))['sha256']
        if hashlib.sha256(weights.read_bytes()).hexdigest() != expected:
            raise ValueError('Trained weights changed since verification')
    return weights
