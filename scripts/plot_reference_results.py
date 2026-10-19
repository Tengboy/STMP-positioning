"""Redraw distributed aggregate reference CSVs, without MAT files or models."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BUNDLE=Path(__file__).resolve().parents[1]
METHODS=['STMP','ML-AWGN','ML-Averaged Ruu','MVALSE','CRB']
COLORS=['#1763b6','#b57808','#a9369e','#c43c39','#222222']
MARKERS=['o','d','s','v',None]

def read_csv(path):
    with open(path,encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))

def draw_iterations(iterations,view,variant):
    fig,ax=plt.subplots(figsize=(6.8,4.7),constrained_layout=True)
    for index,(color,kappa) in enumerate(zip(('#2463a6','#378241','#8c42a2'),(0,5,10))):
        points=sorted([r for r in iterations if r['view']==view and int(r['kappa_db'])==kappa],key=lambda r:int(r['display_point']))
        x=[int(r['display_point']) for r in points]
        for field,style,label in [('stmp_rmse_m','-o','STMP'),('ml_rmse_m','--','ML proxy (update 1)' if view=='legacy_view' else 'ML-AWGN')]:
            ax.semilogy(x,[float(r[field]) for r in points],style,color=color,markersize=4,label=f'{label}, kappa={kappa} dB')
        failed=int(points[-1]['failed_final']) if points else 0
        if view=='all_trials' and failed:
            trials=int(points[-1]['trials'])
            ax.text(.02,.98-.06*index,f'kappa={kappa}: {failed}/{trials} final failures; full RMSE nonfinite',
                    transform=ax.transAxes,va='top',fontsize=7,color=color)
    ax.set(xlabel='Historical displayed point (init + first 9 updates)' if view=='legacy_view' else 'Update number (0 = grid initialization)',ylabel='Position RMSE (m)')
    ax.grid(True,which='both',alpha=.25);ax.legend(fontsize=7,ncol=2,loc='best')
    ax.set_title('CDL-C, SNR=20 dB / '+variant+'\n'+('Final-error mask, conditional results' if view=='legacy_view' else 'All 3000 trials'),fontsize=10)
    return fig

def redraw(source,output,variant):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    tag='fresh_'+variant
    rows=read_csv(source/(tag+'_summary.csv'))
    iterations=read_csv(source/(tag+'_iterations.csv'))
    def save(fig,name):
        for ext in ('png','pdf'):fig.savefig(output/(name+'.'+ext),dpi=300,bbox_inches='tight')
        plt.close(fig)
    def panel(ax,scenario,masked):
        for index,method in enumerate(METHODS):
            points=sorted([r for r in rows if r['scenario']==scenario and float(r['kappa_db'])==0 and r['method']==method],key=lambda r:float(r['snr_db']))
            x=[float(r['snr_db']) for r in points];y=[float(r['legacy_mask_rmse' if masked else 'rmse_full']) for r in points]
            ax.semilogy(x,y,color=COLORS[index],marker=MARKERS[index],linestyle='--' if method=='CRB' else '-',label='MDVALSE' if method=='MVALSE' else method)
            bad=[str(int(a)) for a,b in zip(x,y) if not np.isfinite(b)]
            if bad:ax.text(.02,.98-.07*index,method+': nonfinite at '+','.join(bad)+' dB',transform=ax.transAxes,va='top',fontsize=7)
        ax.set(xlabel='SNR (dB)',ylabel='Position RMSE (m)',xticks=[0,4,8,12,16,20]);ax.grid(True,which='both',alpha=.25)
    for masked in (False,True):
        view='legacy_masks' if masked else 'all_trials'
        title='Historical masks: conditional results' if masked else 'All 3000 trials per point'
        fig,ax=plt.subplots(figsize=(6.2,4.4),constrained_layout=True)
        panel(ax,'CDLC',masked);ax.legend(fontsize=9);ax.set_title('CDL-C / '+variant+'\n'+title,fontsize=10);save(fig,'E1_'+view)
        fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharey=True,constrained_layout=True)
        for ax,scenario in zip(axes,('CDLB','CDLA')):panel(ax,scenario,masked);ax.set_title(scenario)
        axes[0].legend(fontsize=8);fig.suptitle('Frozen C-trained models / '+variant+' / '+title,fontsize=10);save(fig,'E3_'+view)
    for view in ('all_trials','legacy_view'):
        save(draw_iterations(iterations,view,variant),'E2_'+view)
    power=np.loadtxt(source/'E0_power_db.csv',delimiter=',')
    fig,ax=plt.subplots(figsize=(6.4,3.9),constrained_layout=True)
    im=ax.imshow(power,origin='lower',extent=[-1,1,-1,1],aspect='auto',interpolation='nearest',cmap='jet')
    ax.set(xlabel='Normalized horizontal spatial frequency',ylabel='Normalized vertical spatial frequency',title='CDL-C interference, seed 1')
    fig.colorbar(im,ax=ax,label='Unnormalized FFT power (dB)');save(fig,'E0_angular_spectrum')
    training=read_csv(source/'training_validation.csv');meta=json.loads((source/'metadata.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for ax,order in zip(axes,(1,2)):
        points=[r for r in training if int(r['order'])==order]
        for variant_name,color in [('raw','#999999'),('ema','#1464a0')]:
            ax.plot([int(r['step'])/1000 for r in points],[float(r[variant_name+'_loss']) for r in points],color=color,label=variant_name.upper())
        selected=meta['models'][str(order)]['selection']
        ax.scatter(selected['step']/1000,selected['loss'],marker='*',s=80,color='#d35225',label='Frozen selection',zorder=4)
        ax.set(title=f'Order {order}: fixed C validation',xlabel='Training updates (thousands)',ylabel='Validation loss');ax.legend();ax.grid(alpha=.2)
    save(fig,'training_validation')
    return len(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=BUNDLE/'results/reference')
    ap.add_argument('--output',type=Path,default=BUNDLE/'work/reference_plots')
    ap.add_argument('--variant',choices=['historical','paper'],default='historical')
    args=ap.parse_args();count=redraw(args.source,args.output,args.variant)
    print(f'Redrew {count} aggregate rows to {args.output}; no new simulation was performed.')
if __name__=='__main__':main()
