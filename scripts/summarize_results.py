"""Local reproducible summaries and plots; incomplete/failed trials are never zero-filled."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat,savemat
from numerics import summarize_errors
from run_support import DEFAULT_ROOT

SNR=[0,4,8,12,16,20]
METHODS=['STMP','ML-AWGN','ML-Averaged Ruu','MVALSE','CRB']
E2_KAPPAS=[0,5,10]
DISPLAY_NAMES={'MVALSE':'MDVALSE'}

def scenarios():
    return [(scenario,0,snr) for scenario in ('CDLC','CDLB','CDLA') for snr in SNR]+[('CDLC',kappa,20) for kappa in (-5,5,10)]

def summarize(raw,tag,allow_partial=False,expected_count=3000):
    rows=[]; cases={}; masks={}; model_hashes=None
    for scenario,kappa,snr in scenarios():
        case=f'{scenario}_k{kappa}_s{snr}'; path=raw/f'{tag}_{case}.mat'
        if not path.exists():
            if allow_partial: continue
            raise FileNotFoundError(path)
        data=loadmat(path); completed=data['completed'].reshape(-1).astype(bool)
        if not allow_partial and (len(completed)!=expected_count or not completed.all()): raise ValueError('Incomplete case: '+case)
        config=json.loads(str(data['config_json'].item()))
        pair=(config['models']['first'],config['models']['second'])
        if model_hashes is not None and pair!=model_hashes: raise ValueError('Model identity varies across experiments')
        model_hashes=pair
        columns={'STMP':data['squared_error'][:,-1],
                 'ML-AWGN':data['ml_awgn_squared_error'].reshape(-1),
                 'ML-Averaged Ruu':data['ml_cov_squared_error'].reshape(-1),
                 'CRB':data['crb_position_mse'].reshape(-1)}
        if kappa==0:
            baseline_path=raw/f'MVALSE_{case}.mat'
            if baseline_path.exists():
                baseline=loadmat(baseline_path); base_completed=baseline['status'].reshape(-1)>0
                if not np.array_equal(baseline['trial_id'].reshape(-1),data['trial_id'].reshape(-1)):
                    raise ValueError('Baseline trial identities differ')
                if str(baseline['observation_sha256'].item())!=config['observation_sha256']:
                    raise ValueError('Baseline observation identity differs')
                if not allow_partial and not base_completed.all(): raise ValueError('Incomplete MVALSE')
                columns['MVALSE']=baseline['squared_error'].reshape(-1)
            elif not allow_partial: raise FileNotFoundError(baseline_path)
        cases[case]={'data':data,'completed':completed,'columns':columns,'scenario':scenario,'snr':snr,'kappa':kappa,'curves':{}}
        for method,errors in columns.items():
            done=base_completed if method=='MVALSE' else completed
            full=summarize_errors(errors,done)
            finite=done & np.isfinite(errors)
            legacy=done.copy()
            # draw3 plots unfiltered A; the A<=2 arrays are computed but not plotted.
            if scenario in ('CDLC','CDLB') and method in ('STMP','ML-AWGN'): legacy &= errors<=1
            masks[case+'_'+method.replace(' ','_')]=legacy
            legacy_rmse=float(np.sqrt(np.mean(errors[legacy]))) if legacy.any() else float('nan')
            row={'variant':tag,'case':case,'scenario':scenario,'kappa_db':kappa,'snr_db':snr,'method':method,**full,
                 'finite_p50_m':float(np.quantile(np.sqrt(errors[finite]),.5)) if finite.any() else np.nan,
                 'finite_p95_m':float(np.quantile(np.sqrt(errors[finite]),.95)) if finite.any() else np.nan,
                 'finite_p99_m':float(np.quantile(np.sqrt(errors[finite]),.99)) if finite.any() else np.nan,
                 'legacy_mask_count':int(legacy.sum()),'legacy_mask_rmse':legacy_rmse,
                 'rmse_status':'failure_in_full_population' if full['failed'] else 'partial' if full['pending'] else 'complete'}
            rows.append(row); cases[case]['curves'][method]=(full['rmse_full'],legacy_rmse)
        if scenario=='CDLC' and snr==20:
            proxy_mask=completed & (data['squared_error'][:,0]<=1)
            ids=data['trial_id'].reshape(-1)
            key=case+'_E2_ML_proxy_update1'
            masks[key]=proxy_mask
            masks[key+'_retained_trial_ids']=ids[proxy_mask]
            masks[key+'_excluded_trial_ids']=ids[completed & ~proxy_mask]
            cases[case]['legacy_ml_proxy_mask']=proxy_mask
    return rows,cases,masks,model_hashes

def draw(cases,output,tag,e2_only=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    colors={'STMP':'#1763b6','ML-AWGN':'#b57808','ML-Averaged Ruu':'#a9369e','MVALSE':'#c43c39','CRB':'#222222'}
    markers={'STMP':'o','ML-AWGN':'d','ML-Averaged Ruu':'s','MVALSE':'v','CRB':None}
    def save(fig,name):
        for suffix in ('png','pdf'): fig.savefig(output/(name+'.'+suffix),dpi=300,bbox_inches='tight')
        plt.close(fig)
    def snr_plot(ax,scenario,methods,masked):
        for method in methods:
            x=[]; y=[]; failures=[]
            for snr in SNR:
                case=cases.get(f'{scenario}_k0_s{snr}')
                if case is None or method not in case['curves']: continue
                x.append(snr); value=case['curves'][method][int(masked)]; y.append(value)
                if not np.isfinite(value): failures.append(str(snr))
            ax.semilogy(x,y,color=colors[method],marker=markers[method],linestyle='--' if method=='CRB' else '-',label=DISPLAY_NAMES.get(method,method))
            if failures: ax.text(.02,.98-.07*methods.index(method),method+': nonfinite at '+','.join(failures)+' dB',transform=ax.transAxes,va='top',fontsize=7)
        ax.set(xlabel='SNR (dB)',ylabel='Position RMSE (m)',xticks=SNR)
        ax.grid(True,which='both',alpha=.25)
    if not e2_only:
        for masked in (False,True):
            suffix='legacy_masks' if masked else 'all_trials'
            fig,ax=plt.subplots(figsize=(6.2,4.4),constrained_layout=True)
            snr_plot(ax,'CDLC',METHODS,masked); ax.legend(fontsize=9)
            ax.set_title(('Reference masks: not an unconditional comparison' if masked else 'CDL-C, all trials')+'\n'+tag,fontsize=10)
            save(fig,'E1_'+suffix)
            fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharey=True,constrained_layout=True)
            for ax,scenario in zip(axes,('CDLB','CDLA')):
                snr_plot(ax,scenario,METHODS,masked); ax.set_title(scenario)
            axes[0].legend(fontsize=8)
            fig.suptitle(('Historical plotting masks (A unfiltered)' if masked else 'Frozen C-trained models, all trials')+' / '+tag,fontsize=10)
            save(fig,'E3_'+suffix)
    for legacy_view in (False,True):
        fig,ax=plt.subplots(figsize=(6.8,4.7),constrained_layout=True)
        for panel_row,(color,kappa) in enumerate(zip(('#2463a6','#378241','#8c42a2'),E2_KAPPAS)):
            item=cases.get(f'CDLC_k{kappa}_s20')
            if item is None: continue
            data=item['data']; done=item['completed']; errors=data['squared_error']
            trajectory=np.column_stack([data['initial_squared_error'].reshape(-1),errors])
            if legacy_view:
                mask=done & (errors[:,-1]<=1)
                curve=np.sqrt(np.mean(trajectory[mask,:10],axis=0)) if mask.any() else np.full(10,np.nan)
                x=np.arange(1,11); ml=errors[:,0]; valid=item['legacy_ml_proxy_mask']
                ml_value=float(np.sqrt(ml[valid].mean())) if valid.any() else np.nan
            else:
                curve=np.sqrt(np.mean(trajectory[done],axis=0)); x=np.arange(11)
                ml_value=item['curves']['ML-AWGN'][0]
            ax.semilogy(x,curve,'-o',color=color,markersize=4,label=f'STMP, kappa={kappa} dB')
            ml_label='ML proxy (update 1)' if legacy_view else 'ML-AWGN'
            ax.semilogy(x,np.repeat(ml_value,len(x)),'--',color=color,label=f'{ml_label}, kappa={kappa} dB')
            failed=int(np.sum(done & ~np.isfinite(errors[:,-1])))
            if failed and not legacy_view:
                ax.text(.02,.98-.055*panel_row,f'kappa={kappa}: {failed}/{int(done.sum())} final failures; full RMSE nonfinite',
                        transform=ax.transAxes,va='top',fontsize=7,color=color)
        ax.set(xlabel='Historical displayed point (init + first 9 updates)' if legacy_view else 'Update number (0 = grid initialization)',
               ylabel='Position RMSE (m)',xticks=np.arange(1,11) if legacy_view else np.arange(11))
        ax.grid(True,which='both',alpha=.25); ax.legend(fontsize=7,ncol=2)
        ax.set_title(('Reference: final-error mask and historical indexing' if legacy_view else 'CDL-C, 20 dB, all trials')+'\n'+tag,fontsize=10)
        save(fig,'E2_legacy_view' if legacy_view else 'E2_all_trials')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tag',required=True); ap.add_argument('--allow-partial',action='store_true'); ap.add_argument('--no-plots',action='store_true')
    ap.add_argument('--root',default=str(DEFAULT_ROOT)); ap.add_argument('--expected-count',type=int,default=3000)
    args=ap.parse_args(); root=Path(args.root).resolve()
    rows,cases,masks,models=summarize(root/'results/raw',args.tag,args.allow_partial,args.expected_count)
    if not rows: raise ValueError('No actual result files found')
    summary=root/'results/summary'; summary.mkdir(parents=True,exist_ok=True)
    with (summary/(args.tag+'_summary.csv')).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(summary/(args.tag+'_legacy_masks.npz'),**masks)
    curves={}
    for key,item in cases.items():
        curves[key+'_squared_error']=item['data']['squared_error']
        curves[key+'_initial_squared_error']=item['data']['initial_squared_error']
        curves[key+'_completed']=item['completed']
        if 'legacy_ml_proxy_mask' in item:
            mask=item['legacy_ml_proxy_mask']
            proxy=item['data']['squared_error'][:,0]
            curves[key+'_E2_ML_proxy_update1_rmse']=float(np.sqrt(proxy[mask].mean())) if mask.any() else np.nan
            curves[key+'_E2_ML_proxy_update1_mask']=mask
            curves[key+'_trial_id']=item['data']['trial_id'].reshape(-1)
        for method,values in item['curves'].items(): curves[key+'_'+method.replace('-','_').replace(' ','_')]=np.asarray(values)
    savemat(str(summary/(args.tag+'_curves.mat')),curves,oned_as='column')
    metadata={'tag':args.tag,'cases':len(cases),'expected_cases':21,'allow_partial':args.allow_partial,'model_sha256':models,
              'display_method_names':DISPLAY_NAMES,'E2_display_kappa_db':E2_KAPPAS,'E2_omitted_kappa_db':[-5],
              'E2_omission_reason':'User requested on2026-10-18 to omit kappa=-5 from iteration figures; original data and all-case statistics are retained.',
              'mask_warning':'method-specific historical masks are diagnostic, not fair unconditional comparisons',
              'E2_proxy_mask_export':'legacy_masks.npz contains separate <case>_E2_ML_proxy_update1 masks and retained/excluded_trial_ids; curves.mat includes the plotted proxy RMSE, mask and trial IDs',
              'E2_warning':'legacy view selects on final MSE<=1 and plots initialization plus first9 updates; its ML proxy is the first STMP refinement and differs from 2D-initialized ML-AWGN under the literal 1D route; source trajectories retain all10 updates',
              'truth_stop':'main figures use fixed10 updates; truth-dependent stopping is not used'}
    (summary/(args.tag+'_metadata.json')).write_text(json.dumps(metadata,indent=2))
    if not args.no_plots: draw(cases,root/'figures'/args.tag,args.tag)
    print(json.dumps(metadata),flush=True)

if __name__=='__main__': main()
