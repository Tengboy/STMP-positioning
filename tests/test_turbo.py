import sys
import textwrap
import unittest
from pathlib import Path
import numpy as np
import torch
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'scripts')); sys.path.insert(0,str(root/'tests/reference'))
import loc_utils_refine as historical
from localization import ArrayGeometry
from turbo import turbo_estimate,gaussian_extrinsic

class GaussianScore(torch.nn.Module):
    def __init__(self,order): super().__init__(); self.order=order
    def forward(self,x,std):
        denominator=.5+std[:,None,None,None]**2
        return -x/denominator if self.order==1 else -torch.ones_like(x)/denominator

class TurboTests(unittest.TestCase):
    def test_extrinsic_lmmse_identity(self):
        h=torch.tensor([1.+.3j]); g=torch.tensor([.4-.2j]); y=torch.tensor([2.+1.j])
        vh=torch.tensor([.03]); vg=torch.tensor([.7]); noise=.2
        residual=y-h-g
        post=h+vh/(vh+vg+noise)*residual
        postvar=vh*(vg+noise)/(vh+vg+noise)
        mean,variance=gaussian_extrinsic(post,postvar,h,vh)
        torch.testing.assert_close(mean,y-g)
        torch.testing.assert_close(variance,vg+noise)

    def test_turbo_matches_extracted_original_updates(self):
        torch.set_num_threads(2); torch.manual_seed(93)
        array=ArrayGeometry(8,4)
        position=torch.tensor([[10.,18.,1.5]])
        y=array.steering(array.mu_from_position(position))+.1*torch.randn(1,4,8,dtype=torch.complex64)
        first=GaussianScore(1); second=GaussianScore(2)
        syspar={'BS_pos':np.array([0.,0.,25.]),'scaling':1.,'UE_posZ':1.5,'Nx':8,'Ny':4,
                'BS_ex':np.array([1,0,0]),'BS_ey':np.array([0,0,1]),'noise_pow':.1}
        initial=historical.LoS_angle_iniest_tensor(y[0],syspar)
        env=dict(torch=torch,np=np,locf=historical,Ite=3,ksi=1e-6,damping=1.,Nx=8,Ny=4,
            H_B_pri_mean=y[0],H_B_pri_var=torch.tensor(.1),UE_pos_mean=initial,
            H_LoS_t=torch.zeros_like(y[0]),H_NLoS_t=torch.zeros_like(y[0]),y_noise_t=y[0],
            pow_noise=.1,sysPar=syspar,device='cpu',s1_model=first,s2_model=second,
            W=torch.cat([torch.eye(4,dtype=torch.complex64)]*2,dim=1),trajectory=[])
        class Config: device='cpu'
        env['config']=Config()
        source=(root/'tests/reference/turbo_updates.py.txt').read_text(encoding='utf-8')
        exec(compile(source,'<historical-update-excerpt>','exec'),env)
        actual=turbo_estimate(y,.1,first,second,array,iterations=3,initialization='2d',variance_convention='legacy_mean_real')
        expected=torch.stack(env['trajectory'])
        torch.testing.assert_close(actual['position'][0],expected,rtol=2e-4,atol=2e-3)
        torch.testing.assert_close(actual['posterior_variance'][0,-1],env['G_C_post_var'],rtol=1e-5,atol=1e-6)

    def test_posterior_real_imag_variance(self):
        torch.manual_seed(4); array=ArrayGeometry(8,4)
        y=array.steering(array.mu_from_position(torch.tensor([[10.,18.,1.5]])))
        out=turbo_estimate(y,.1,GaussianScore(1),GaussianScore(2),array,iterations=1,initialization='2d',variance_convention='complex_sum')
        self.assertAlmostEqual(float(out['posterior_variance'][0,0]),.5,places=6)

if __name__=='__main__': unittest.main()
