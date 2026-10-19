"""Explicit scientific conventions shared by audit and evaluation tooling."""
import numpy as np

def complex_covariance(channels, order='F'):
    h=np.asarray(channels)
    if h.ndim!=3 or len(h)==0 or order not in ('F','C'):
        raise ValueError('Expected nonempty [samples,vertical,horizontal] channels and F/C order')
    vectors=np.stack([sample.reshape(-1,order=order) for sample in h])
    return vectors.T @ vectors.conj()/len(vectors)

def posterior_from_scores(ri, std_real, first, second, variance_convention):
    y=np.asarray(ri)
    if y.ndim!=4 or y.shape[1]!=2:
        raise ValueError('Expected [batch,2,vertical,horizontal]')
    if np.shape(first)!=y.shape or np.shape(second)!=y.shape:
        raise ValueError('Score shapes must equal the RI input')
    sigma=np.asarray(std_real)
    if sigma.ndim==1: sigma=sigma[:,None,None,None]
    mean=y+sigma**2*first
    variance_ri=sigma**4*second+sigma**2
    if variance_convention=='complex_sum':
        variance=variance_ri.sum(axis=1).mean(axis=(1,2))
    elif variance_convention=='legacy_mean_real':
        variance=variance_ri.mean(axis=(1,2,3))
    else:
        raise ValueError('Unknown variance convention')
    return mean,variance

def summarize_errors(errors, completed):
    errors=np.asarray(errors,dtype=np.float64)
    completed=np.asarray(completed,dtype=bool)
    if errors.shape!=completed.shape: raise ValueError('Status shape differs from errors')
    selected=errors[completed]
    finite=selected[np.isfinite(selected)]
    if np.any(finite<0): raise ValueError('Squared errors must be nonnegative')
    failed=int(np.sum(~np.isfinite(selected)))
    return {'completed':int(completed.sum()),'pending':int((~completed).sum()),'failed':failed,
            'rmse_full':float('inf') if failed else float(np.sqrt(np.mean(selected))) if selected.size else float('nan'),
            'rmse_finite_subset':float(np.sqrt(np.mean(finite))) if finite.size else float('nan')}
