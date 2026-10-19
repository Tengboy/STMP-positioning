"""Fixed-iteration B -> C -> A STMP; truth is not an input to inference."""
import torch

def broadcast(value,array):
    return value.reshape((-1,)+(1,)*(array.ndim-1))

def gaussian_extrinsic(post_mean,post_var,prior_mean,prior_var):
    variance=1/(1/post_var-1/prior_var)
    mean=broadcast(variance,post_mean)*(post_mean/broadcast(post_var,post_mean)-prior_mean/broadcast(prior_var,post_mean))
    return mean,variance

def network_score(model,ri,std):
    parameter=next(model.parameters(),None)
    dtype=parameter.dtype if parameter is not None else ri.dtype
    return model(ri.to(dtype),std.to(dtype)).to(ri.dtype)

@torch.no_grad()
def turbo_estimate(y,noise_variance,first,second,array,iterations=10,initialization='1d',variance_convention='complex_sum',damping=1.):
    if variance_convention not in ('complex_sum','legacy_mean_real'): raise ValueError('Variance convention required')
    if not 0<=damping<=1: raise ValueError('Invalid damping')
    batch=len(y)
    noise=torch.full((batch,),float(noise_variance),dtype=array.dtype,device=y.device)
    position=array.initialize(y,initialization)
    initial=position.clone()
    h_prior=y.clone(); h_var=noise.clone()
    positions=[]; variances=[]; negative=[]; clipped=[]; ext_variances=[]; prior_variances=[]
    for iteration in range(iterations):
        position,amplitude=array.refine(h_prior,position,h_var)
        h_post=array.steering(array.mu_from_position(position),amplitude)
        # Exact Gaussian division, written without cancellation of close precisions.
        h_ext_var=h_var/(array.n-1)
        h_ext=(array.n*h_post-h_prior)/(array.n-1)
        if iteration==0:
            g_prior=y-h_ext; g_var=torch.ones_like(noise)
        else:
            g_prior=damping*g_a_ext+(1-damping)*g_prior
            g_var=damping*g_a_ext_var+(1-damping)*g_var
        ri=torch.stack([g_prior.real,g_prior.imag],dim=1)
        std=(g_var.abs()/2).sqrt()
        first_score=network_score(first,ri,std); second_score=network_score(second,ri,std)
        posterior=ri+std[:,None,None,None]**2*first_score
        g_post=torch.complex(posterior[:,0],posterior[:,1])
        raw_variance=std[:,None,None,None]**4*second_score+std[:,None,None,None]**2
        negative.append((raw_variance<0).to(array.dtype).mean(dim=(1,2,3)))
        clipped.append((raw_variance<1e-6).to(array.dtype).mean(dim=(1,2,3)))
        g_post_var=raw_variance.clamp(min=1e-6).mean(dim=(1,2,3))
        if variance_convention=='complex_sum': g_post_var=2*g_post_var
        g_ext,g_ext_var=gaussian_extrinsic(g_post,g_post_var,g_prior,g_var)
        if iteration==0:
            h_a=h_ext; h_a_var=h_ext_var; g_a=g_ext; g_a_var=g_ext_var
        else:
            h_a=damping*h_ext+(1-damping)*h_a
            h_a_var=damping*h_ext_var+(1-damping)*h_a_var
            g_a=damping*g_ext+(1-damping)*g_a
            g_a_var=damping*g_ext_var+(1-damping)*g_a_var
        # Algebraic result of the original LMMSE posterior followed by Gaussian division.
        h_a_ext=y-g_a; h_a_ext_var=g_a_var+noise
        g_a_ext=y-h_a; g_a_ext_var=h_a_var+noise
        h_prior=damping*h_a_ext+(1-damping)*h_prior
        h_var=damping*h_a_ext_var+(1-damping)*h_var
        positions.append(position.clone()); variances.append(g_post_var.clone())
        ext_variances.append(g_ext_var.clone()); prior_variances.append(g_var.clone())
    return {'initial_position':initial,'position':torch.stack(positions,dim=1),
            'posterior_variance':torch.stack(variances,dim=1),
            'prior_variance':torch.stack(prior_variances,dim=1),
            'extrinsic_variance':torch.stack(ext_variances,dim=1),
            'negative_RI_fraction':torch.stack(negative,dim=1),
            'clipped_RI_fraction':torch.stack(clipped,dim=1)}
