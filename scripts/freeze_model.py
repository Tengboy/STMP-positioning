"""Freeze the minimum fixed C-validation checkpoint after training has stopped."""
import argparse, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
from artifact_support import atomic_json, sha256, verify_validation_selection
from run_support import DEFAULT_ROOT, dataset_identity, frozen_model

def parameter_hash(state):
    digest=hashlib.sha256()
    for name,tensor in state.items():
        digest.update(name.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default=str(DEFAULT_ROOT))
    ap.add_argument('--order',type=int,choices=[1,2],required=True)
    ap.add_argument('--inspect',action='store_true')
    args=ap.parse_args();root=Path(args.root).resolve();work=root/f'runs/order{args.order}'
    record_path=root/f'configs/order{args.order}_frozen.json'
    if record_path.exists():
        _,record=frozen_model(root,args.order); print(json.dumps(record,indent=2));return
    config=json.loads((work/'config.json').read_text(encoding='utf-8'))
    status=json.loads((work/'status.json').read_text(encoding='utf-8'))
    datasets={split:dataset_identity(config['arguments'][split],split) for split in ('train','val')}
    if datasets!=config['datasets']: raise ValueError('Training or validation dataset changed')
    metrics=[json.loads(line) for line in (work/'metrics.jsonl').read_text().splitlines() if line.strip()]
    import torch
    source=work/'best.pth';source_hash=sha256(source)
    state=torch.load(source,map_location='cpu',weights_only=True)
    selected=verify_validation_selection(metrics,status,state,expected_count=datasets['val']['count'])
    if state['order']!=args.order or config['arguments']['order']!=args.order: raise ValueError('Order differs')
    if args.order==2:
        _,teacher=frozen_model(root,1)
        if state['teacher_sha256']!=teacher['sha256'] or status['teacher_parameters_sha256']!=teacher['parameters_sha256']:
            raise ValueError('Frozen teacher identity was not preserved')
    if args.inspect:
        print(json.dumps({'selection':selected,'training_status':status,'datasets':datasets},indent=2));return
    model_dir=root/'models';model_dir.mkdir(exist_ok=True)
    target=model_dir/f'order{args.order}_step{state["step"]}_{state["weight_variant"]}_{source_hash[:12]}.pth'
    payload={k:state[k] for k in ('model','step','order','weight_variant','best_metric','teacher_sha256')}
    tensor_hash=parameter_hash(payload['model'])
    if target.exists():
        if parameter_hash(torch.load(target,map_location='cpu',weights_only=True)['model'])!=tensor_hash:
            raise ValueError('Existing export differs')
    else:
        temp=target.with_suffix('.partial');torch.save(payload,temp);os.replace(temp,target)
    if sha256(source)!=source_hash: raise ValueError('Training checkpoint changed during export')
    record={'order':args.order,'checkpoint':target.relative_to(root).as_posix(),'sha256':sha256(target),
            'parameters_sha256':tensor_hash,'selection':selected,'datasets':datasets,'training_status':status,
            'teacher_sha256':state['teacher_sha256'],'source_sha256':source_hash,
            'evidence_sha256':{n:sha256(work/n) for n in ('config.json','status.json','metrics.jsonl')},
            'selection_rule':'minimum raw/EMA loss across all fixed C validation events; no test/transfer selection',
            'frozen_utc':datetime.now(timezone.utc).isoformat()}
    atomic_json(record_path,record);print(json.dumps(record,indent=2))
if __name__=='__main__': main()
