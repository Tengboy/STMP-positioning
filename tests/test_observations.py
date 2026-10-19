import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).absolute().parents[1]/'scripts'))
from observation_support import steering_from_position,position_from_mu,make_trial

class ObservationTests(unittest.TestCase):
    def test_geometry_and_vectorization(self):
        position=np.array([8.,17.,1.5])
        signal,mu=steering_from_position(position,np.exp(.7j))
        np.testing.assert_allclose(position_from_mu(mu),position,atol=1e-12)
        expected=np.exp(.7j)*np.kron(np.exp(1j*np.pi*np.arange(64)*mu[0]),np.exp(1j*np.pi*np.arange(16)*mu[1]))
        np.testing.assert_allclose(signal.flatten(order='F'),expected,atol=1e-13)
        self.assertAlmostEqual(float(np.mean(abs(signal)**2)),1.)

    def test_shared_observation_and_noise_scaling(self):
        h=np.full((16,64),1j)
        low=make_trial(h,2,0,20)
        high=make_trial(h,2,10,20)
        np.testing.assert_array_equal(low['UE_pos'],high['UE_pos'])
        np.testing.assert_allclose(high['noise'],low['noise']*np.sqrt(10),atol=1e-14)
        np.testing.assert_allclose(low['Y'],low['H_LoS']+low['H_NLoS']+low['noise'],atol=1e-14)
        self.assertEqual(low['noise_variance'],.01)
        self.assertEqual(low['trial_id'],2)

if __name__=='__main__': unittest.main()
