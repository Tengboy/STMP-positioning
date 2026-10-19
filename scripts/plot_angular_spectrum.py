"""Local E0 figure from one newly generated CDL-C interference realization."""
import argparse
import json
from run_support import DEFAULT_ROOT
from pathlib import Path
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from make_observations import sha256

ap=argparse.ArgumentParser(); ap.add_argument('--root',default=str(DEFAULT_ROOT))
args=ap.parse_args(); root=Path(args.root).resolve()
for sub in ('figures','results/summary','reports'): (root/sub).mkdir(parents=True,exist_ok=True)
source=root/'datasets/CDLC_test.mat'
with h5py.File(source,'r') as f:
    ri=np.asarray(f['h_f_test'][...,0]).transpose(2,1,0)
    seed=int(np.asarray(f['seed_ids']).flat[0])
h=ri[0].astype(float)+1j*ri[1].astype(float)
assert h.shape==(16,64)
spectrum=np.fft.fftshift(np.fft.fft2(h))
power_db=10*np.log10(np.maximum(abs(spectrum)**2,1e-12))
mu_x=2*np.fft.fftshift(np.fft.fftfreq(64)); mu_y=2*np.fft.fftshift(np.fft.fftfreq(16))
fig,ax=plt.subplots(figsize=(6.4,3.9),constrained_layout=True)
im=ax.imshow(power_db,origin='lower',extent=[-1,1,-1,1],aspect='auto',interpolation='nearest',cmap='jet')
ax.set(xlabel='Normalized horizontal spatial frequency',ylabel='Normalized vertical spatial frequency')
ax.set_xticks([-1,-.5,0,.5,1]); ax.set_yticks([-1,-.5,0,.5,1])
fig.colorbar(im,ax=ax,label=r'$10\log_{10}|\mathrm{FFT}_2(z)|^2$ (dB)')
ax.set_title('CDL-C interference, seed %d'%seed,fontsize=11)
for extension in ('png','pdf'): fig.savefig(root/('figures/E0_angular_spectrum.'+extension),dpi=300)
plt.close(fig)
savemat(str(root/'results/summary/E0_angular_spectrum.mat'),dict(channel=h,spectrum=spectrum,power_db=power_db,
    horizontal_fft_bins=mu_x,vertical_fft_bins=mu_y,seed=seed,channel_sha256=sha256(source)))
metadata={'dataset':str(source),'dataset_sha256':sha256(source),'seed':seed,'shape':[16,64],
          'formula':'10 log10(abs(fftshift(fft2(H)))^2); unnormalized FFT; display extent [-1,1]^2 follows original plot',
          'mean_element_power':float(np.mean(abs(h)**2)),
          'note':'Paper realization seed is unspecified; this figure reproduces the operation and structure, not identical pixels.'}
(root/'reports/E0_metadata.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps(metadata),flush=True)
