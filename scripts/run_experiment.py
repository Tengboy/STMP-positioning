"""Run one documented stage; all generated artifacts remain under --root."""
import argparse, json, subprocess, sys
from pathlib import Path
from run_support import BUNDLE, DEFAULT_ROOT, frozen_model

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['observations','covariance','train1','freeze1','train2','freeze2','evaluate','plots','e0'])
    ap.add_argument('--root',default=str(DEFAULT_ROOT))
    ap.add_argument('--variant',choices=['historical','paper'],default='historical')
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--count',type=int,default=3000)
    ap.add_argument('--batch-size',type=int)
    ap.add_argument('--max-steps',type=int)
    ap.add_argument('--max-hours',type=float)
    ap.add_argument('--eval-every',type=int,default=1000)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--inspect',action='store_true')
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args();root=Path(a.root).resolve();tag='fresh_'+a.variant
    if a.count<1:ap.error('--count must be positive')
    def run(script,*args):
        command=[sys.executable,str(BUNDLE/'scripts'/script),*map(str,args)]
        print(json.dumps(command),flush=True)
        if not a.dry_run:subprocess.run(command,check=True)
    if a.stage=='observations':
        for scenario in ('CDLC','CDLB','CDLA'):
            run('make_observations.py','--data',root/f'datasets/{scenario}_test.mat','--scenario',scenario,'--out-dir',root/'observations','--count',a.count)
        run('make_observations.py','--data',root/'datasets/CDLC_test.mat','--scenario','CDLC','--out-dir',root/'observations','--count',a.count,'--kappa',-5,5,10,'--snr',20)
    elif a.stage=='covariance':
        run('make_covariance.py','--data',root/'datasets/CDLC_train.mat','--output',root/'cache/covariance_C.npz','--device',a.device)
    elif a.stage.startswith('train'):
        order=int(a.stage[-1]);extra=[]
        if order==2:
            teacher,_=frozen_model(root,1);extra=['--teacher',teacher]
        if a.resume:extra+=['--resume']
        run('train_score.py','--root',root,'--order',order,'--train',root/'datasets/CDLC_train.mat','--val',root/'datasets/CDLC_val.mat',
            '--workdir',root/f'runs/order{order}','--max-steps',a.max_steps or (100001 if order==1 else 141128),
            '--max-hours',a.max_hours or (12 if order==1 else 20),'--batch-size',a.batch_size or 256,
            '--eval-every',a.eval_every,'--device',a.device,*extra)
    elif a.stage.startswith('freeze'):
        run('freeze_model.py','--root',root,'--order',a.stage[-1],*(['--inspect'] if a.inspect else []))
    elif a.stage=='evaluate':
        first,_=frozen_model(root,1);second,_=frozen_model(root,2)
        settings=json.loads((BUNDLE/'configs/experiments.json').read_text())
        variant=settings['variants'][a.variant]
        files=[root/'observations'/name for name in settings['observation_order']]
        run('evaluate.py','--root',root,'--first',first,'--second',second,'--covariance',root/'cache/covariance_C.npz',
            '--out-dir',root/'results/raw','--tag',tag,'--initialization',variant['initialization'],
            '--variance-convention',variant['variance'],'--covariance-convention',variant['covariance'],
            '--batch-size',a.batch_size or 64,'--iterations',10,'--count',a.count,'--device',a.device,'--observations',*files)
    elif a.stage=='plots':
        run('summarize_results.py','--root',root,'--tag',tag,'--expected-count',a.count)
    elif a.stage=='e0':run('plot_angular_spectrum.py','--root',root)
if __name__=='__main__': main()
