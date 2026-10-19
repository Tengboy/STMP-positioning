"""Validation and file-identity rules shared by freezing and collection tools."""
import hashlib
import json
import math
import os
import re
from pathlib import Path, PurePosixPath


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def checked_relative(root, relative):
    """Accept portable, regular artifact paths, never traversal or temporary files."""
    value = str(relative)
    path = PurePosixPath(value)
    if (not re.fullmatch(r'[A-Za-z0-9_./-]+', value) or path.is_absolute()
            or '..' in path.parts or '.' in value.split('/') or str(path) != value
            or path.suffix in ('.partial', '.tmp')):
        raise ValueError('Invalid artifact path: ' + value)
    target = Path(root).joinpath(*path.parts)
    current = Path(root)
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Symlink artifact path: ' + value)
    return target


def verify_validation_selection(metrics, status, checkpoint, expected_count=2000):
    if (status.get('state') != 'training_finished' or not metrics
            or metrics[-1].get('event') != 'finished'
            or metrics[-1]['step'] != status['step']):
        raise ValueError('Training has not finished or has resumed')
    candidates = []
    for row in metrics:
        if row.get('event') != 'validation':
            continue
        if row['count'] != expected_count:
            raise ValueError('Validation population changed')
        for variant in ('raw', 'ema'):
            loss = float(row[variant + '_loss'])
            if not math.isfinite(loss):
                raise ValueError('Nonfinite validation metric')
            candidates.append({'step': row['step'], 'variant': variant, 'loss': loss,
                               'count': row['count'], 'diagnostics': row[variant + '_diagnostics']})
    if not candidates:
        raise ValueError('No validation evidence')
    # The runner updates best only on strict improvement, so ties retain the first.
    chosen = min(candidates, key=lambda row: row['loss'])
    if (checkpoint['step'] != chosen['step'] or checkpoint['weight_variant'] != chosen['variant']
            or not math.isclose(float(checkpoint['best_metric']), chosen['loss'], rel_tol=1e-12)):
        raise ValueError('Best checkpoint does not match fixed validation selection')
    if (not math.isclose(float(status['best_metric']), chosen['loss'], rel_tol=1e-12)
            or status['best_variant'] != chosen['variant']):
        raise ValueError('Finished status does not match validation selection')
    chosen['validation_events'] = len(candidates) // 2
    chosen['recent_validation'] = [row for row in metrics if row.get('event') == 'validation'][-8:]
    return chosen
