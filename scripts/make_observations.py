"""Create immutable MATLAB/Python-shared observations from completed new channels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import h5py
import numpy as np
from scipy.io import savemat
from observation_support import make_trial,SNR_GRID

def sha256(path):
    value=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''): value.update(block)
    return value.hexdigest()

def generate(data_path,scenario,out_dir,count=None,kappas=(0,),snrs=SNR_GRID):
    data_path=Path(data_path); out_dir=Path(out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    with h5py.File(data_path,'r') as f:
        ri=np.array(f['h_f_test']).transpose(3,2,1,0)
        if int(np.asarray(f['completed']).flat[0])!=len(ri): raise ValueError('Incomplete channels')
        channel_seeds=np.asarray(f['seed_ids']).reshape(-1)
    if ri.shape[1:]!=(2,16,64) or not np.isfinite(ri).all(): raise ValueError('Invalid channel schema')
    count=len(ri) if count is None else int(count)
    if not 0<count<=len(ri): raise ValueError('Invalid observation count')
    indices=np.random.RandomState(12345).permutation(len(ri))[:count]
    data_hash=sha256(data_path)
    records=[]
    for kappa in kappas:
        for snr in snrs:
            tag='%s_k%g_s%g'%(scenario.replace('-',''),kappa,snr)
            target=out_dir/(tag+'.mat'); marker=out_dir/(tag+'.json')
            if marker.exists():
                existing=json.loads(marker.read_text())
                if existing['channel_sha256']!=data_hash or existing['count']!=count:
                    raise ValueError('Existing observations use different inputs')
                if sha256(target)!=existing['sha256']: raise ValueError('Existing observation hash differs')
                records.append(existing); continue
            values={key:np.empty((count,16,64),dtype=np.complex128) for key in ('H_NLoS','H_LoS','noise','Y')}
            values.update(UE_pos=np.empty((count,3)),rho=np.empty(count,dtype=np.complex128),mu=np.empty((count,2)))
            for trial,index in enumerate(indices):
                sample=make_trial(ri[index,0]+1j*ri[index,1],trial,kappa,snr)
                for key in values: values[key][trial]=sample[key]
            values.update(trial_id=np.arange(count,dtype=np.int64),channel_index=indices,
                channel_seed=channel_seeds[indices],snr_db=float(snr),kappa_db=float(kappa),
                noise_variance=10.**((kappa-snr)/10.),scenario=scenario,
                channel_sha256=data_hash,vectorization='F physical; array shape N,Ny,Nx',
                indexing='zero-based trial_id and channel_index',schema_version=1)
            partial=target.with_suffix('.partial')
            savemat(str(partial),values,appendmat=False,do_compression=False,oned_as='column')
            os.replace(partial,target)
            record={'experiment_id':tag,'path':str(target),'sha256':sha256(target),'bytes':target.stat().st_size,
                    'channel_sha256':data_hash,'scenario':scenario,'count':count,'snr_db':snr,'kappa_db':kappa}
            marker.write_text(json.dumps(record,indent=2))
            records.append(record)
            print(json.dumps(record),flush=True)
    return records

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True); ap.add_argument('--scenario',required=True)
    ap.add_argument('--out-dir',required=True); ap.add_argument('--count',type=int)
    ap.add_argument('--kappa',type=float,nargs='+',default=[0])
    ap.add_argument('--snr',type=float,nargs='+',default=SNR_GRID)
    a=ap.parse_args()
    generate(a.data,a.scenario,a.out_dir,a.count,a.kappa,a.snr)
