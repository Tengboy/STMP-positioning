"""Resumable batched STMP/ML/CRB evaluation from immutable shared MAT observations."""
import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.io import loadmat,savemat
import torch
from make_observations import sha256
from model_runtime import load_pair,backend_identity,network_identity
from run_support import DEFAULT_ROOT
from localization import ArrayGeometry,position_crb
from turbo import turbo_estimate

def error(estimate,truth):
    result=np.sum((estimate-truth)**2,axis=-1)
    result[~np.isfinite(result)]=np.inf
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default=str(DEFAULT_ROOT))
    ap.add_argument('--first',required=True); ap.add_argument('--second',required=True)
    ap.add_argument('--covariance',required=True); ap.add_argument('--observations',nargs='+',required=True)
    ap.add_argument('--out-dir',required=True); ap.add_argument('--tag',required=True)
    ap.add_argument('--initialization',choices=['1d','2d'],required=True)
    ap.add_argument('--variance-convention',choices=['complex_sum','legacy_mean_real'],required=True)
    ap.add_argument('--covariance-convention',choices=['correct','legacy_conjugate'],required=True)
    ap.add_argument('--batch-size',type=int,default=64); ap.add_argument('--count',type=int)
    ap.add_argument('--iterations',type=int,default=10)
    ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args(); root=Path(args.root).resolve(); outdir=Path(args.out_dir).resolve()
    if args.batch_size<1 or args.iterations<1 or (args.count is not None and args.count<1): raise ValueError('Counts must be positive')
    if root not in outdir.parents: raise ValueError('Output must be inside reproduction directory')
    outdir.mkdir(parents=True,exist_ok=True)
    first,second=load_pair(root,args.first,args.second,args.device)
    geometry=ArrayGeometry(device=args.device)
    hashes={'first':sha256(args.first),'second':sha256(args.second),'covariance':sha256(args.covariance)}
    code_hashes={name:sha256(Path(__file__).parent/name) for name in ('evaluate.py','localization.py','turbo.py','model_runtime.py','run_support.py','artifact_support.py')}
    code_version=hashlib.sha256(json.dumps(code_hashes,sort_keys=True).encode()).hexdigest()
    snapshot=root/'source_snapshots'/'evaluation_versions'/code_version
    snapshot.mkdir(parents=True,exist_ok=True)
    for name in code_hashes:
        if not (snapshot/name).exists(): shutil.copy2(Path(__file__).parent/name,snapshot/name)
    with np.load(args.covariance) as f:
        covariance=f['covariance']; covariance_source=str(f['channel_sha256'])
    if args.covariance_convention=='legacy_conjugate': covariance=covariance.conj()
    covariance=torch.as_tensor(covariance.copy(),dtype=torch.complex64,device=args.device)
    runtime_backend=backend_identity(args.device)
    network_sources=network_identity()
    inverse_cache={}
    for observation in args.observations:
        started=time.monotonic(); data=loadmat(observation)
        count=len(data['Y']) if args.count is None else min(args.count,len(data['Y']))
        noise=float(data['noise_variance'].item()); kappa=float(data['kappa_db'].item()); snr=float(data['snr_db'].item())
        config={'tag':args.tag,'observation_sha256':sha256(observation),'models':hashes,'source_hashes':code_hashes,
                'backend':runtime_backend,'network_sources':network_sources,
                'count':count,'initialization':args.initialization,'variance_convention':args.variance_convention,
                'covariance_convention':args.covariance_convention,'iterations':args.iterations,
                'covariance_source_sha256':covariance_source,'damping':1.,'torch_version':str(torch.__version__),
                'batch_size':args.batch_size,'geometry_dtype':'float32','network_dtype':'float32',
                'cudnn_tf32':torch.backends.cudnn.allow_tf32,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
                'cudnn_benchmark':torch.backends.cudnn.benchmark,'code_version':code_version}
        config_text=json.dumps(config,sort_keys=True)
        config_digest=hashlib.sha256(config_text.encode()).hexdigest()
        target=outdir/(args.tag+'_'+Path(observation).stem+'.mat')
        if target.exists():
            values=loadmat(target)
            if str(values['config_digest'].item())!=config_digest: raise ValueError('Existing result configuration changed: '+str(target))
            values={key:value for key,value in values.items() if not key.startswith('__')}
            completed=values['completed'].reshape(-1).astype(bool)
        else:
            values={'squared_error':np.full((count,args.iterations),np.nan),
                'initial_squared_error':np.full(count,np.nan),'ml_awgn_squared_error':np.full(count,np.nan),
                'ml_cov_squared_error':np.full(count,np.nan),'position':np.full((count,args.iterations,3),np.nan),
                'initial_position':np.full((count,3),np.nan),'ml_awgn_position':np.full((count,3),np.nan),
                'ml_cov_position':np.full((count,3),np.nan),'posterior_variance':np.full((count,args.iterations),np.nan),
                'prior_variance':np.full((count,args.iterations),np.nan),'extrinsic_variance':np.full((count,args.iterations),np.nan),
                'negative_RI_fraction':np.full((count,args.iterations),np.nan),'clipped_RI_fraction':np.full((count,args.iterations),np.nan),
                'seconds_per_trial':np.full(count,np.nan),'crb_position_mse':position_crb(data['UE_pos'][:count],noise,10**(kappa/10)),
                'UE_pos':data['UE_pos'][:count],'trial_id':data['trial_id'][:count],
                'channel_index':data['channel_index'][:count],'snr_db':snr,'kappa_db':kappa,
                'status':np.zeros(count,dtype=np.uint8),'config_json':config_text,'config_digest':config_digest,
                'status_legend':'0 pending; 1 all finite STMP iterates; 2 nonfinite iterate (errors retained as Inf)'}
            completed=np.zeros(count,dtype=bool)
        # MATLAB-v5 restores vector fields as columns; normalize only per-trial scalar arrays.
        for key in ('initial_squared_error','ml_awgn_squared_error','ml_cov_squared_error','seconds_per_trial','crb_position_mse','status'):
            values[key]=np.asarray(values[key]).reshape(-1)
        pending=np.flatnonzero(~completed)
        if len(pending) and noise not in inverse_cache:
            inverse_cache[noise]=torch.linalg.inv(covariance+noise*torch.eye(1024,dtype=torch.complex64,device=args.device))
        for offset in range(0,len(pending),args.batch_size):
            indices=pending[offset:offset+args.batch_size]
            tick=time.monotonic(); y=torch.as_tensor(data['Y'][indices],dtype=torch.complex64,device=args.device)
            with torch.no_grad():
                initial2d=geometry.initialize(y,'2d')
                variance=torch.full((len(y),),noise,device=y.device)
                awgn,_=geometry.refine(y,initial2d,variance)
                averaged,_=geometry.refine(y,initial2d,variance,inverse_cache[noise])
                result=turbo_estimate(y,noise,first,second,geometry,args.iterations,args.initialization,args.variance_convention)
            arrays={key:value.cpu().numpy() for key,value in result.items()}
            truth=data['UE_pos'][indices]
            for key,array in arrays.items(): values[key][indices]=array
            values['squared_error'][indices]=error(arrays['position'],truth[:,None,:])
            values['initial_squared_error'][indices]=error(arrays['initial_position'],truth)
            values['ml_awgn_position'][indices]=awgn.cpu().numpy(); values['ml_cov_position'][indices]=averaged.cpu().numpy()
            values['ml_awgn_squared_error'][indices]=error(awgn.cpu().numpy(),truth)
            values['ml_cov_squared_error'][indices]=error(averaged.cpu().numpy(),truth)
            values['status'][indices]=np.where(np.isfinite(values['squared_error'][indices]).all(axis=1),1,2)
            values['seconds_per_trial'][indices]=(time.monotonic()-tick)/len(indices)
            completed[indices]=True; values['completed']=completed.astype(np.uint8)
            temporary=target.with_suffix('.partial')
            savemat(str(temporary),values,appendmat=False,do_compression=False,oned_as='column'); os.replace(temporary,target)
            selected=values['squared_error'][completed,-1]
            print(json.dumps({'event':'evaluation','file':target.name,'completed':int(completed.sum()),'count':count,
                'failed':int((~np.isfinite(selected)).sum()),'full_rmse':float(np.sqrt(selected.mean())) if np.isfinite(selected).all() else None,
                'elapsed_seconds':time.monotonic()-started}),flush=True)
        print(json.dumps({'event':'evaluation_finished','file':str(target),'sha256':sha256(target),'completed':int(completed.sum())}),flush=True)

if __name__=='__main__': main()
