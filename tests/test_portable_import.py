import os, subprocess, sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class PortableImportTests(unittest.TestCase):
    def test_cpu_network_does_not_require_cuda_toolchain(self):
        code="import torch; from models.ncsnpp import NCSNpp; from configs.ve.ffhq_256_ncsnpp_continuous import get_config; torch.set_num_threads(2); c=get_config(); c.model.nf=32; m=NCSNpp(c); x=torch.zeros(1,2,16,64); y=m(x,torch.ones(1)); assert y.shape==x.shape and torch.isfinite(y).all()"
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',CUDA_HOME='',PYTHONDONTWRITEBYTECODE='1')
        env['PYTHONPATH']=str(ROOT/'third_party/score_sde/sde_1st')+os.pathsep+os.environ.get('PYTHONPATH','')
        run=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
if __name__=='__main__': unittest.main()
