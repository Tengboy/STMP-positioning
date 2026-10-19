"""Commit validation history and selected weights with each recovery checkpoint."""
import hashlib
import os
import uuid
from pathlib import Path
import torch


def recovery_snapshot(work):
    work = Path(work)
    history = (work/'metrics.jsonl').read_bytes()
    best = work/'best.pth'
    return {
        'metrics_size': len(history),
        'metrics_sha256': hashlib.sha256(history).hexdigest(),
        'best_checkpoint': torch.load(best, map_location='cpu', weights_only=True) if best.exists() else None,
    }


def recover_history(work, state):
    """Rollback only evidence after the committed checkpoint; retain it for audit."""
    work = Path(work)
    path = work/'metrics.jsonl'
    history = path.read_bytes()
    size = state['metrics_size']
    prefix = history[:size]
    if len(history) < size or hashlib.sha256(prefix).hexdigest() != state['metrics_sha256']:
        raise ValueError('Committed validation history is missing or changed')
    if not prefix.endswith(b'\n'):
        raise ValueError('Committed validation history is not a complete JSONL prefix')
    tail = history[size:]
    best = work/'best.pth'
    if tail:
        archive = work/'recovery_history'
        archive.mkdir(exist_ok=True)
        token = uuid.uuid4().hex
        (archive/(token+'.jsonl')).write_bytes(tail)
        if best.exists():
            import shutil
            shutil.copy2(best, archive/(token+'_uncommitted_best.pth'))
        temporary = path.with_suffix('.recovery.partial')
        temporary.write_bytes(prefix)
        os.replace(temporary, path)
    if state['best_checkpoint'] is None:
        if best.exists():
            archive = work/'recovery_history'
            archive.mkdir(exist_ok=True)
            os.replace(best, archive/(uuid.uuid4().hex+'_uncommitted_best.pth'))
    else:
        temporary = best.with_suffix('.recovery.partial')
        torch.save(state['best_checkpoint'], temporary)
        os.replace(temporary, best)
