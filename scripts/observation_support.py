"""Physical array geometry and shared observations, in double precision."""
import numpy as np

BS=np.array([0.,0.,25.])
SNR_GRID=[0,4,8,12,16,20]

def steering_from_position(position,rho=1.,nx=64,ny=16):
    direction=np.asarray(position,dtype=float)-BS
    direction/=np.linalg.norm(direction)
    mu=direction[[0,2]]
    ax=np.exp(1j*np.pi*np.arange(nx)*mu[0])
    ay=np.exp(1j*np.pi*np.arange(ny)*mu[1])
    return rho*ay[:,None]*ax[None,:],mu

def position_from_mu(mu):
    mx,my=map(float,mu)
    if not np.isfinite([mx,my]).all() or mx*mx+my*my>=1 or my>=0:
        raise ValueError('Direction does not intersect the forward UE plane')
    direction=np.array([mx,np.sqrt(1-mx*mx-my*my),my])
    return BS+(1.5-BS[2])/my*direction

def make_trial(h,trial_id,kappa_db,snr_db):
    if snr_db not in SNR_GRID: raise ValueError('SNR must be in the frozen experiment grid')
    rng=np.random.RandomState(int(trial_id)*1000+42)
    theta=np.deg2rad(rng.rand()*120+30)
    distance=10+rng.rand()*20
    position=np.array([distance*np.cos(theta),distance*np.sin(theta),1.5])
    rho=np.exp(1j*rng.uniform(0,2*np.pi))
    los,mu=steering_from_position(position,rho)
    h=np.asarray(h,dtype=np.complex128)
    h=h/np.sqrt(np.mean(abs(h)**2))
    # Always consume the full historical SNR sequence, including for E2-only points.
    for current in SNR_GRID:
        standard=(rng.randn(*h.shape)+1j*rng.randn(*h.shape))/np.sqrt(2)
        if current==snr_db: noise_standard=standard
    kappa=10.**(kappa_db/10.)
    variance=kappa/10.**(snr_db/10.)
    noise=np.sqrt(variance)*noise_standard
    return {'trial_id':int(trial_id),'UE_pos':position,'rho':rho,'mu':mu,
            'H_NLoS':h,'H_LoS':los,'noise':noise,
            'Y':np.sqrt(kappa)*los+h+noise,'noise_variance':variance}
