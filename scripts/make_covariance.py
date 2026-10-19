"""Source-domain covariance E[h h^H], in the evaluator's C-order coordinates."""
import argparse
import json
import os
from pathlib import Path
import h5py
import numpy as np
import torch
from make_observations import sha256

def build(data_path,output,device='cuda:0'):
    torch.set_num_threads(2)
    output=Path(output)
    digest=sha256(data_path)
    if output.exists():
        with np.load(output) as old:
            if str(old['channel_sha256'])!=digest: raise ValueError('Existing covariance uses another dataset')
        print('COVARIANCE_ALREADY_VERIFIED',flush=True); return
    with h5py.File(data_path,'r') as f:
        dataset=f['h_f_train']; count=dataset.shape[-1]
        if 'completed' in f and int(np.asarray(f['completed']).flat[0])!=count: raise ValueError('Incomplete train data')
        covariance=torch.zeros((1024,1024),dtype=torch.complex128,device=device)
        for offset in range(0,count,1024):
            ri=np.array(dataset[...,offset:offset+1024]).transpose(3,2,1,0)
            h=torch.as_tensor(ri[:,0]+1j*ri[:,1],device=device,dtype=torch.complex128).reshape(len(ri),-1)
            covariance+=h.T@h.conj()
    covariance=(covariance/count).cpu().numpy()
    if not np.isfinite(covariance).all(): raise ValueError('Invalid covariance')
    np.testing.assert_allclose(covariance,covariance.conj().T,rtol=1e-12,atol=1e-12)
    temporary=output.with_suffix('.partial')
    output.parent.mkdir(parents=True,exist_ok=True)
    with open(temporary,'wb') as file:
        np.savez(file,covariance=covariance,channel_sha256=digest,count=count,
                 convention='E[h h^H]; C-order flatten; legacy matrix is complex conjugate')
    os.replace(temporary,output)
    print(json.dumps({'output':str(output),'count':count,'mean_power':float(np.trace(covariance).real/1024),
                      'sha256':sha256(output),'channel_sha256':digest}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--output',required=True)
    ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args(); build(args.data,args.output,args.device)
