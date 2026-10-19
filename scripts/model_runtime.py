import codecs
import os
import sys
from pathlib import Path
from run_support import BUNDLE,code_path
from artifact_support import sha256

def backend_identity(device):
    """Prevent a partial result from silently mixing hardware/backend arithmetic."""
    import platform
    import torch
    device=torch.device(device)
    identity={'device_type':device.type,'device':str(device),
              'torch_version':str(torch.__version__),'platform':platform.system(),
              'machine':platform.machine(),'processor':platform.processor()}
    if device.type=='cuda':
        available=torch.cuda.is_available()
        identity.update(cuda_runtime=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),
                        cuda_available=available)
        if available:
            identity.update(device_name=torch.cuda.get_device_name(device),
                            capability=list(torch.cuda.get_device_capability(device)))
    return identity


def network_identity(bundle=None):
    """Both inference networks use the second-order NCSN++ entry point."""
    bundle=Path(bundle) if bundle is not None else BUNDLE
    code=bundle/'third_party/score_sde/sde_2nd'
    paths=[bundle/'configs/order2.json']
    paths.extend(p for p in sorted(code.rglob('*')) if p.suffix in ('.py','.cu','.cpp'))
    return {p.relative_to(bundle).as_posix():sha256(p) for p in paths}


def load_pair(root,first_path,second_path,device='cuda:0'):
    root=Path(root)
    os.environ['TORCH_EXTENSIONS_DIR']=str(root/'cache/torch_extensions')
    os.environ['MPLCONFIGDIR']=str(root/'cache/matplotlib')
    os.environ['MAX_JOBS']='2'
    sys.path.insert(0,str(code_path(2)))
    import numpy as np
    import torch
    from configs.ve.ffhq_256_ncsnpp_continuous import get_config
    from models.ncsnpp import NCSNpp
    def scalar_compat(dtype,value): return np.core.multiarray.scalar(dtype,value)
    scalar_compat.__module__='numpy._core.multiarray'; scalar_compat.__name__='scalar'
    torch.serialization.add_safe_globals([scalar_compat,codecs.encode,np.dtype,
        type(np.dtype('float64')),type(np.dtype('float32')),type(np.dtype('int64'))])
    torch.set_num_threads(2)
    config=get_config(); config.model.nf=128; config.device=torch.device(device)
    models=[]
    for path in (first_path,second_path):
        model=NCSNpp(config).to(config.device)
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        model.load_state_dict({k.removeprefix('module.'):v for k,v in checkpoint['model'].items()},strict=True)
        model.eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        models.append(model)
        del checkpoint
    return models
