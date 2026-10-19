"""Portable paths and immutable source-domain training identities."""
import json
from pathlib import Path
from artifact_support import sha256
BUNDLE=Path(__file__).resolve().parents[1]
DEFAULT_ROOT=BUNDLE/'work'

def code_path(order):
    return BUNDLE/'third_party/score_sde'/('sde_1st' if order==1 else 'sde_2nd')

def dataset_identity(path, split):
    import h5py
    import numpy as np
    with h5py.File(path,'r') as f:
        def word(key): return ''.join(chr(int(x)) for x in np.asarray(f[key]).ravel())
        if word('scenario')!='CDL-C' or word('split')!=split:
            raise ValueError('Training/selection requires the CDL-C '+split+' split')
        key='h_f_train' if split=='train' else 'h_f_test'
        shape=f[key].shape
        if len(shape)!=4 or shape[:3]!=(64,16,2) or shape[-1]<1:
            raise ValueError('Invalid channel dimensions')
        count=shape[-1];seeds=np.asarray(f['seed_ids']).reshape(-1)
        if int(np.asarray(f['completed']).flat[0])!=count or len(seeds)!=count:
            raise ValueError('Incomplete channel dataset')
        if len(np.unique(seeds))!=count:
            raise ValueError('Duplicate channel seeds')
        if (split=='val' and not np.all((seeds>=3001)&(seeds<=5000))) or (split=='train' and not np.all(seeds>=5001)):
            raise ValueError('C train/validation seed ranges must remain disjoint from test seeds')
    return {'sha256':sha256(path),'count':int(count),'scenario':'CDL-C','split':split}

def check_resume(previous,current):
    if previous!=current: raise ValueError('Resume inputs, seed, batch, device, or code changed; use a new work directory')

def frozen_model(root,order):
    root=Path(root)
    record=json.loads((root/f'configs/order{order}_frozen.json').read_text(encoding='utf-8'))
    path=root/record['checkpoint']
    if sha256(path)!=record['sha256']: raise ValueError('Frozen model integrity failure')
    return path,record
