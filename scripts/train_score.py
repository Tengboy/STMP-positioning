"""Bounded training around the copied original architecture and loss functions."""
import argparse
import codecs
import hashlib
import json
import os
import random
import shutil
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from artifact_support import atomic_json, sha256
from run_support import DEFAULT_ROOT, code_path, dataset_identity, check_resume, frozen_model

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default=str(DEFAULT_ROOT))
    ap.add_argument('--order',type=int,choices=[1,2],required=True)
    ap.add_argument('--train',required=True); ap.add_argument('--val',required=True)
    ap.add_argument('--workdir',required=True); ap.add_argument('--teacher')
    ap.add_argument('--max-steps',type=int,required=True)
    ap.add_argument('--max-hours',type=float,required=True)
    ap.add_argument('--batch-size',type=int,default=256)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--eval-every',type=int,default=1000)
    ap.add_argument('--deadline-utc',default=None)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    root=Path(args.root).resolve(); work=Path(args.workdir).resolve()
    if root not in work.parents: raise ValueError('Workdir must be inside reproduction root')
    if args.order==2 and not args.teacher: raise ValueError('Second order requires a frozen teacher')
    if args.max_steps<1 or args.max_hours<=0 or args.batch_size<1 or args.eval_every<1:
        raise ValueError('Training limits and batch size must be positive')
    if (root/f'configs/order{args.order}_frozen.json').exists():
        raise ValueError('This order is frozen; train a new run instead of mutating its teacher/model')
    started=time.monotonic()
    deadline=datetime.fromisoformat(args.deadline_utc).timestamp() if args.deadline_utc else float('inf')
    os.environ['TORCH_EXTENSIONS_DIR']=str(root/'cache/torch_extensions')
    os.environ['MPLCONFIGDIR']=str(root/'cache/matplotlib')
    os.environ['MAX_JOBS']='2'
    code=code_path(args.order)
    sys.path.insert(0,str(code))
    import h5py
    import numpy as np
    import torch
    from train_support import BatchStream,freeze_teacher,optimizer_update
    from checkpoint_support import recovery_snapshot,recover_history
    from configs.ve.ffhq_256_ncsnpp_continuous import get_config
    from models.ncsnpp import NCSNpp
    from models.ema import ExponentialMovingAverage
    import losses
    from sde_lib import VESDE

    def scalar_compat(dtype,value): return np.core.multiarray.scalar(dtype,value)
    scalar_compat.__module__='numpy._core.multiarray'; scalar_compat.__name__='scalar'
    torch.serialization.add_safe_globals([scalar_compat,codecs.encode,np.dtype,
        type(np.dtype('float64')),type(np.dtype('float32')),type(np.dtype('int64'))])
    torch.set_num_threads(2)
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    work.mkdir(parents=True,exist_ok=True)
    checkpoint=work/'last.pth'
    if not args.resume and any((work/name).exists() for name in ('last.pth','metrics.jsonl','best.pth')):
        raise FileExistsError('Use --resume for an existing run, or a new work directory if no recovery checkpoint exists')
    cfg=get_config()
    cfg.model.nf=128
    cfg.device=torch.device(args.device)
    cfg.training.batch_size=args.batch_size
    cfg.training.n_iters=args.max_steps
    cfg.data.path_train=args.train; cfg.data.path_test=args.val
    cfg.optim.lr=2e-4 if args.order==1 else 5e-5
    cfg.seed=args.seed
    data_identity={'train':dataset_identity(args.train,'train'),'val':dataset_identity(args.val,'val')}
    source_hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                   for name in ('train_score.py','train_support.py','run_support.py','checkpoint_support.py','artifact_support.py')}
    source_version=hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()
    source_snapshot=root/'source_snapshots'/'training_versions'/source_version
    source_snapshot.mkdir(parents=True,exist_ok=True)
    for name in source_hashes:
        if not (source_snapshot/name).exists(): shutil.copy2(Path(__file__).parent/name,source_snapshot/name)
    contract={'datasets':data_identity,'order':args.order,'seed':args.seed,'batch_size':args.batch_size,
              'device':str(cfg.device),'sources':source_hashes,
              'network_sources':{str(p.relative_to(code)):sha256(p) for p in sorted(code.rglob('*')) if p.suffix in ('.py','.cu','.cpp')},
              'model':cfg.model.to_dict(),'optim':cfg.optim.to_dict()}
    if args.resume: check_resume(json.loads((work/'resume_contract.json').read_text()),contract)
    atomic_json(work/'resume_contract.json',contract)
    if args.order==2:
        teacher_path,teacher_record=frozen_model(root,1)
        if Path(args.teacher).resolve()!=teacher_path.resolve(): raise ValueError('Teacher must be the frozen first-order export')
    (work/'config.json').write_text(json.dumps({'arguments':vars(args),
        'model':cfg.model.to_dict(),'optim':cfg.optim.to_dict(),
        'training':cfg.training.to_dict(),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'runner_sources':source_hashes,'runner_version':source_version,
        'torch_version':torch.__version__,'device':torch.cuda.get_device_name(cfg.device) if cfg.device.type=='cuda' else 'cpu',
        'datasets':data_identity,
        'started_utc':datetime.now(timezone.utc).isoformat()},indent=2,default=str))

    def load_data(path,key):
        with h5py.File(path,'r') as f:
            value=np.asarray(f[key],dtype=np.float32).transpose(3,2,1,0)
            if 'completed' in f and int(np.asarray(f['completed']).flat[0])!=len(value):
                raise ValueError('Dataset generation is incomplete')
        if value.shape[1:]!=(2,16,64) or not np.isfinite(value).all():
            raise ValueError('Invalid channel schema/values')
        return torch.from_numpy(np.ascontiguousarray(value))

    train_data=load_data(args.train,'h_f_train')
    val_data=load_data(args.val,'h_f_test')
    model=NCSNpp(cfg).to(cfg.device)
    optimizer=losses.get_optimizer(cfg,model.parameters())
    ema=ExponentialMovingAverage(model.parameters(),decay=cfg.model.ema_rate)
    def parameter_hash(network):
        digest=hashlib.sha256()
        for name,tensor in network.state_dict().items():
            digest.update(name.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    teacher=None; teacher_hash=None; teacher_parameters_hash=None
    if args.order==2:
        teacher=NCSNpp(cfg).to(cfg.device)
        state=torch.load(args.teacher,map_location='cpu',weights_only=True)
        teacher.load_state_dict({k.removeprefix('module.'):v for k,v in state['model'].items()},strict=True)
        freeze_teacher(teacher)
        teacher_hash=hashlib.sha256(Path(args.teacher).read_bytes()).hexdigest()
        teacher_parameters_hash=parameter_hash(teacher)
        del state
    step=0; elapsed_before=0.; best_metric=float('inf'); best_variant=None
    stream_state=None
    if args.resume:
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if state['order']!=args.order or state['teacher_sha256']!=teacher_hash:
            raise ValueError('Resume order/teacher changed')
        recover_history(work,state['recovery'])
        model.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer']); ema.load_state_dict(state['ema'])
        ema.shadow_params=[tensor.to(cfg.device) for tensor in ema.shadow_params]
        step=int(state['step']); elapsed_before=float(state['elapsed_seconds'])
        best_metric=float(state['best_metric']); best_variant=state['best_variant']
        stream_state=state['batch_stream']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng']) if cfg.device.type=='cuda' else None
        del state
    stream=BatchStream(train_data,args.batch_size,args.seed+100,stream_state)
    sde=VESDE(cfg.model.sigma_min,cfg.model.sigma_max,cfg.model.num_scales)
    train_loss=losses.get_sde_loss_fn(sde,True,reduce_mean=False,continuous=True,likelihood_weighting=False)
    eval_loss=losses.get_sde_loss_fn(sde,False,reduce_mean=False,continuous=True,likelihood_weighting=False)
    stopped=[False]
    signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stopped.__setitem__(0,True))
    log=(work/'metrics.jsonl').open('a',encoding='utf-8',newline='\n',buffering=1)

    def emit(event,**fields):
        row={'event':event,'step':step,'utc':datetime.now(timezone.utc).isoformat(),**fields}
        line=json.dumps(row,allow_nan=False)
        print(line,flush=True); log.write(line+'\n')

    def evaluate(limit):
        total=0.; count=0
        with torch.random.fork_rng(devices=[cfg.device.index or 0] if cfg.device.type=='cuda' else []):
            torch.manual_seed(20261015); torch.cuda.manual_seed_all(20261015)
            with torch.no_grad():
                for offset in range(0,min(limit,len(val_data)),args.batch_size):
                    batch=val_data[offset:min(offset+args.batch_size,limit)].to(cfg.device)
                    loss=eval_loss(model,batch) if teacher is None else eval_loss(teacher,model,batch)
                    total+=float(loss.item())*len(batch); count+=len(batch)
        return total/count

    def save(path,variant='raw'):
        payload={'model':model.state_dict(),'step':step,'order':args.order,
                 'teacher_sha256':teacher_hash,'weight_variant':variant,'best_metric':best_metric}
        if path.name!='best.pth':
            log.flush(); os.fsync(log.fileno())
            payload.update(optimizer=optimizer.state_dict(),ema=ema.state_dict(),
                batch_stream=stream.state_dict(),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if cfg.device.type=='cuda' else [],
                best_variant=best_variant,elapsed_seconds=elapsed_before+time.monotonic()-started,
                recovery=recovery_snapshot(work))
        temp=path.with_suffix('.partial')
        torch.save(payload,temp); os.replace(temp,path)

    def denoising_diagnostics():
        diagnostics=[]
        with torch.random.fork_rng(devices=[cfg.device.index or 0] if cfg.device.type=='cuda' else []),torch.no_grad():
            torch.manual_seed(20261016); torch.cuda.manual_seed_all(20261016)
            clean=val_data[:min(128,len(val_data))].to(cfg.device)
            noise=torch.randn_like(clean)
            for sigma in (.1,.5,1.):
                std=torch.full((len(clean),),sigma,device=cfg.device)
                noisy=clean+sigma*noise
                first=(model if teacher is None else teacher)(noisy,std)
                posterior=noisy+sigma*sigma*first
                row={'sigma_real':sigma,'posterior_nmse':float(torch.sum((posterior-clean)**2)/torch.sum(clean**2))}
                if teacher is not None:
                    variance=sigma**4*model(noisy,std)+sigma**2
                    row.update(negative_RI_variance_fraction=float((variance<0).float().mean()),
                        clipped_RI_variance_fraction=float((variance<1e-6).float().mean()),
                        mean_complex_variance=float(variance.sum(dim=1).mean()))
                diagnostics.append(row)
        return diagnostics

    def validation():
        nonlocal best_metric,best_variant
        # A single fixed validation population prevents selecting on incomparable subsets.
        limit=len(val_data)
        raw=evaluate(limit)
        raw_diagnostics=denoising_diagnostics()
        if not np.isfinite(raw): raise FloatingPointError('Nonfinite raw validation')
        if raw<best_metric:
            best_metric=raw; best_variant='raw'; save(work/'best.pth','raw')
        ema.store(model.parameters()); ema.copy_to(model.parameters())
        try:
            smooth=evaluate(limit)
            ema_diagnostics=denoising_diagnostics()
            if not np.isfinite(smooth): raise FloatingPointError('Nonfinite EMA validation')
            if smooth<best_metric:
                best_metric=smooth; best_variant='ema'; save(work/'best.pth','ema')
        finally: ema.restore(model.parameters())
        emit('validation',raw_loss=raw,ema_loss=smooth,count=limit,best_metric=best_metric,best_variant=best_variant,
             raw_diagnostics=raw_diagnostics,ema_diagnostics=ema_diagnostics)

    atomic_json(work/'status.json',{'state':'training','order':args.order,'step':step})
    emit('started',order=args.order,train_samples=len(train_data),validation_samples=len(val_data),
         max_steps=args.max_steps,max_hours=args.max_hours,teacher_sha256=teacher_hash)
    if not args.resume: save(checkpoint)
    initial_step=step; reason='max_steps'; last_loss=None
    try:
        while step<args.max_steps:
            if stopped[0]: reason='signal'; break
            if time.time()>=deadline: reason='global_deadline'; break
            if elapsed_before+time.monotonic()-started>=args.max_hours*3600:
                reason='stage_time_limit'; break
            batch=stream.next().to(cfg.device)
            closure=(lambda:train_loss(model,batch)) if teacher is None else (lambda:train_loss(teacher,model,batch))
            last_loss=optimizer_update(model,optimizer,closure,step,cfg.optim.lr,cfg.optim.warmup,cfg.optim.grad_clip)
            step+=1; ema.update(model.parameters())
            if step%50==0 or step==1:
                elapsed=time.monotonic()-started
                emit('training',loss=last_loss,elapsed_seconds=elapsed_before+elapsed,
                     seconds_per_update=elapsed/max(step-initial_step,1),
                     peak_cuda_bytes=torch.cuda.max_memory_allocated(cfg.device) if cfg.device.type=='cuda' else 0)
            if step%args.eval_every==0: validation()
            if step%5000==0: save(checkpoint)
            if step%10000==0: save(work/f'step_{step}.pth')
        validation(); save(checkpoint)
        if teacher is not None:
            if sha256(args.teacher)!=teacher_hash: raise ValueError('Teacher file changed during training')
            assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
            if parameter_hash(teacher)!=teacher_parameters_hash: raise AssertionError('Teacher parameters changed')
        status={'state':'training_finished','reason':reason,'step':step,'best_metric':best_metric,
                'best_variant':best_variant,'last_loss':last_loss,
                'elapsed_seconds':elapsed_before+time.monotonic()-started}
        status['teacher_parameters_sha256']=teacher_parameters_hash
        (work/'status.json').write_text(json.dumps(status,indent=2))
        emit('finished',**status)
    except Exception as exc:
        (work/'status.json').write_text(json.dumps({'state':'failed','step':step,'error':repr(exc)},indent=2))
        raise
    finally:
        log.close()

if __name__=='__main__': main()
