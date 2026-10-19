import importlib.util, sys, tempfile, unittest
from pathlib import Path
import h5py, numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
class RunContractTests(unittest.TestCase):
    def test_validation_rejects_target_scenario_and_changed_resume(self):
        self.assertIsNotNone(importlib.util.find_spec('run_support'), 'Portable dataset/resume contract is missing')
        from run_support import dataset_identity, check_resume
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'CDLC_val.mat'
            with h5py.File(p,'w') as f:
                f['scenario']=np.array([ord(x) for x in 'CDL-A'],dtype=np.uint16)
                f['split']=np.array([ord(x) for x in 'val'],dtype=np.uint16)
                f['seed_ids']=[3001,3002];f['completed']=2
                f['h_f_test']=np.zeros((64,16,2,2))
            with self.assertRaisesRegex(ValueError,'CDL-C'):dataset_identity(p,'val')
            with h5py.File(p,'r+') as f:f['scenario'][-1]=ord('C')
            info=dataset_identity(p,'val');self.assertEqual(info['count'],2)
            check_resume({'data':info},{'data':dict(info)})
            with self.assertRaisesRegex(ValueError,'changed'):check_resume({'data':info},{'data':{'sha256':'different'}})
if __name__=='__main__': unittest.main()
