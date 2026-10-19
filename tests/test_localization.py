import sys
import unittest
from pathlib import Path
import numpy as np
import torch
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'scripts'))
sys.path.insert(0,str(root/'tests/reference'))
import loc_utils_refine as historical
from localization import ArrayGeometry,position_crb

class LocalizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(29)
        self.sys={'Nx':8,'Ny':4,'BS_pos':np.array([0.,0.,25.]),'BS_ex':np.array([1,0,0]),
                  'BS_ey':np.array([0,0,1]),'UE_posZ':1.5,'noise_pow':.1,'scaling':1.}

    def test_analytic_crb_matches_full_nuisance_fisher(self):
        position=np.array([10.,18.,1.5])
        params=np.array([1.3,.7,10.,18.])
        expected=historical.CRB_calculate(params,self.sys)
        actual=position_crb(position,.1,1.3**2,nx=8,ny=4)
        self.assertAlmostEqual(float(actual/expected),1.,places=4)

    def test_grid_and_refinement_match_historical(self):
        array=ArrayGeometry(8,4,dtype=torch.float32)
        p=torch.tensor([[10.,18.,1.5]])
        signal=array.steering(array.mu_from_position(p),torch.tensor([1.+.2j]))
        signal=signal+.05*(torch.randn_like(signal.real)+1j*torch.randn_like(signal.real))
        for mode,reference in [('2d',historical.LoS_angle_iniest_tensor),('1d',historical.LoS_angle_iniest_1D)]:
            new=array.initialize(signal,mode)
            old=reference(signal[0],self.sys)
            torch.testing.assert_close(new[0],old,rtol=1e-5,atol=1e-4)
        ini=p+torch.tensor([[.2,-.1,0.]])
        for dense in (False,True):
            covariance=.1*torch.eye(32,dtype=torch.complex64)
            if dense:
                v=torch.randn(32,dtype=torch.complex64)*.01
                covariance+=v[:,None]*v.conj()[None,:]
            precision=torch.linalg.inv(covariance) if dense else None
            new,amp=array.refine(signal,ini,torch.tensor([.1]),precision)
            old,_,oldamp=historical.LoS_angle_est_byhand(signal[0],covariance if dense else .1,ini[0],self.sys,scalar_cov_flag=0 if dense else 1)
            torch.testing.assert_close(new[0],old,rtol=2e-4,atol=2e-3)
            torch.testing.assert_close(amp[0],oldamp.squeeze(),rtol=2e-4,atol=2e-3)

    def test_batch_equivalence_and_gradient_finite_difference(self):
        array=ArrayGeometry(8,4,dtype=torch.float64)
        positions=torch.tensor([[8.,17.,1.5],[-12.,20.,1.5],[1.,12.,1.5]],dtype=torch.float64)
        mu=array.mu_from_position(positions)
        y=array.steering(mu,torch.ones(3,dtype=torch.complex128))+.05*torch.randn(3,4,8,dtype=torch.complex128)
        initial=positions+torch.tensor([.2,.1,0.])
        batch,_=array.refine(y,initial,torch.full((3,),.1))
        serial=torch.cat([array.refine(y[i:i+1],initial[i:i+1],torch.tensor([.1]))[0] for i in range(3)])
        torch.testing.assert_close(batch,serial,rtol=1e-10,atol=1e-10)
        mu=mu+torch.tensor([.01,-.01])
        obj,grad=array.objective_gradient(y,mu,torch.full((3,),.1))
        for axis in range(2):
            delta=torch.zeros_like(mu); delta[:,axis]=1e-6
            plus=array.objective_gradient(y,mu+delta,torch.full((3,),.1))[0]
            minus=array.objective_gradient(y,mu-delta,torch.full((3,),.1))[0]
            torch.testing.assert_close(grad[:,axis],(plus-minus)/2e-6,rtol=1e-7,atol=1e-6)

if __name__=='__main__': unittest.main()
