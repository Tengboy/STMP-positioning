"""Analytic contracts: these catch conjugation, variance and averaging errors."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).absolute().parents[1]/'scripts'))
import numerics

class NumericalContracts(unittest.TestCase):
    def test_covariance_keeps_imaginary_correlation_direction(self):
        h=np.array([[[1,1j]],[[2,2j]]],dtype=np.complex128)
        expected=np.array([[2.5,-2.5j],[2.5j,2.5]])
        np.testing.assert_allclose(numerics.complex_covariance(h,order='F'),expected)

    def test_tweedie_complex_variance_adds_real_and_imaginary_parts(self):
        y=np.array([[[[1.]],[[3.]]]])
        first=-y/3
        second=np.full_like(y,-1/3)
        mean,var=numerics.posterior_from_scores(y,1.,first,second,'complex_sum')
        np.testing.assert_allclose(mean,[[[[2/3]],[[2.]]]])
        np.testing.assert_allclose(var,[4/3])
        _,legacy=numerics.posterior_from_scores(y,1.,first,second,'legacy_mean_real')
        np.testing.assert_allclose(legacy,[2/3])

    def test_rmse_preserves_failures_and_zero_is_valid(self):
        result=numerics.summarize_errors(np.array([0.,9.,np.inf,np.nan]),np.array([True,True,True,False]))
        self.assertEqual(result['completed'],3)
        self.assertEqual(result['failed'],1)
        self.assertEqual(result['pending'],1)
        self.assertTrue(np.isinf(result['rmse_full']))
        self.assertAlmostEqual(result['rmse_finite_subset'],np.sqrt(4.5))

if __name__=='__main__': unittest.main()
