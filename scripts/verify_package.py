"""Verify all shipped files against SOURCE_MANIFEST.json (standard library only)."""
import argparse,hashlib,json
from pathlib import Path

def verify(root):
    root=Path(root).resolve();manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    failed=[]
    for row in manifest['files']:
        path=root/row['path']
        if not path.is_file() or path.is_symlink():failed.append(row['path']);continue
        if path.stat().st_size!=row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:failed.append(row['path'])
    if failed:raise ValueError('Missing or changed release files: '+', '.join(failed))
    print(f"Verified {len(manifest['files'])} shipped files; generated work files are not part of this manifest.")
    return len(manifest['files'])
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    verify(ap.parse_args().root)
