import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.io import savemat
root=Path(__file__).absolute().parents[1]
sys.path.insert(0,str(root/'scripts'))
from summarize_results import summarize

class SummaryTests(unittest.TestCase):
    def test_e2_proxy_exports_its_own_mask_and_trial_ids(self):
        with tempfile.TemporaryDirectory(prefix='summary_test_',dir=str(root/'tests')) as name:
            raw=Path(name).absolute()
            fixture={'squared_error':np.array([[0.,.1],[4.,.1],[np.inf,np.inf],[.1,.1]]),
                'completed':np.array([1,1,1,0],dtype=np.uint8),
                'initial_squared_error':np.zeros(4),'ml_awgn_squared_error':np.zeros(4),
                'ml_cov_squared_error':np.ones(4),'crb_position_mse':np.ones(4),
                'trial_id':np.array([101,205,309,410]),
                'config_json':json.dumps({'models':{'first':'synthetic1','second':'synthetic2'},'observation_sha256':'synthetic'})}
            savemat(str(raw/'test_CDLC_k-5_s20.mat'),fixture,oned_as='column')
            _,_,masks,_=summarize(raw,'test',True)
            np.testing.assert_array_equal(masks['CDLC_k-5_s20_ML-AWGN'],[True,True,True,False])
            np.testing.assert_array_equal(masks['CDLC_k-5_s20_E2_ML_proxy_update1'],[True,False,False,False])
            np.testing.assert_array_equal(masks['CDLC_k-5_s20_E2_ML_proxy_update1_retained_trial_ids'],[101])
            np.testing.assert_array_equal(masks['CDLC_k-5_s20_E2_ML_proxy_update1_excluded_trial_ids'],[205,309])

    def test_failure_retention_and_exact_historical_masks(self):
        with tempfile.TemporaryDirectory(prefix='summary_test_',dir=str(root/'tests')) as name:
            raw=Path(name).absolute()
            self.assertEqual(raw.parent,root/'tests')
            fixture={'squared_error':np.array([[0.],[4.],[np.inf]]),'completed':np.ones(3,dtype=np.uint8),
                'initial_squared_error':np.zeros(3),'ml_awgn_squared_error':np.array([0.,0.,9.]),
                'ml_cov_squared_error':np.array([3.,2.,1.]),'crb_position_mse':np.ones(3),
                'trial_id':np.arange(3),'config_json':json.dumps({'models':{'first':'synthetic1','second':'synthetic2'},'observation_sha256':'synthetic'})}
            for scenario in ('CDLC','CDLA'):
                savemat(str(raw/f'test_{scenario}_k0_s0.mat'),fixture,oned_as='column')
            rows,cases,masks,_=summarize(raw,'test',True)
            stmp=next(row for row in rows if row['scenario']=='CDLC' and row['method']=='STMP')
            self.assertEqual(stmp['failed'],1); self.assertTrue(np.isinf(stmp['rmse_full']))
            self.assertEqual(stmp['completed'],3); self.assertEqual(stmp['legacy_mask_count'],1)
            self.assertEqual(stmp['legacy_mask_rmse'],0.)
            np.testing.assert_array_equal(masks['CDLC_k0_s0_STMP'],[True,False,False])
            np.testing.assert_array_equal(masks['CDLA_k0_s0_STMP'],[True,True,True])
            with self.assertRaises(ValueError): summarize(raw,'test',False)


    def test_transfer_baselines_keep_failures_and_require_identical_observations(self):
        for scenario in ('CDLA','CDLB'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory(prefix='summary_transfer_',dir=str(root/'tests')) as name:
                raw=Path(name).absolute()
                case=scenario+'_k0_s0'
                fixture={'squared_error':np.zeros((3,2)),'completed':np.ones(3,dtype=np.uint8),
                    'initial_squared_error':np.zeros(3),'ml_awgn_squared_error':np.zeros(3),
                    'ml_cov_squared_error':np.zeros(3),'crb_position_mse':np.ones(3),
                    'trial_id':np.array([11,22,33]),
                    'config_json':json.dumps({'models':{'first':'synthetic1','second':'synthetic2'},'observation_sha256':'shared-input'})}
                baseline={'squared_error':np.array([0.,4.,np.inf]),'status':np.array([1,1,3]),
                    'trial_id':np.array([11,22,33]),'observation_sha256':'shared-input'}
                savemat(str(raw/('test_'+case+'.mat')),fixture,oned_as='column')
                savemat(str(raw/('MVALSE_'+case+'.mat')),baseline,oned_as='column')
                rows,_,masks,_=summarize(raw,'test',True)
                matching=[row for row in rows if row['method']=='MVALSE']
                self.assertEqual(len(matching),1,'A/B baseline must be included')
                self.assertEqual(matching[0]['failed'],1)
                self.assertTrue(np.isinf(matching[0]['rmse_full']))
                self.assertEqual(matching[0]['legacy_mask_count'],3)
                np.testing.assert_array_equal(masks[case+'_MVALSE'],[True,True,True])
                baseline['observation_sha256']='different-input'
                savemat(str(raw/('MVALSE_'+case+'.mat')),baseline,oned_as='column')
                with self.assertRaisesRegex(ValueError,'Baseline observation identity differs'):
                    summarize(raw,'test',True)

if __name__=='__main__': unittest.main()
