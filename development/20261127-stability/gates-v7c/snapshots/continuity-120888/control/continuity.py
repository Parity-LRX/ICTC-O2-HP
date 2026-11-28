"""Content signatures for uninterrupted states, without changing tensor values."""
import hashlib,json
import numpy as np
import torch
def signature(value):
    digest=hashlib.sha256()
    def visit(v):
        digest.update((type(v).__name__+'[').encode())
        if isinstance(v,torch.Tensor):
            v=v.detach().cpu().contiguous();digest.update(str((v.dtype,tuple(v.shape))).encode())
            digest.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v,np.ndarray):
            digest.update(str((v.dtype,v.shape)).encode());digest.update(v.tobytes())
        elif isinstance(v,dict):
            for key in sorted(v,key=str):digest.update(repr(key).encode());visit(v[key])
        elif isinstance(v,(list,tuple)):
            digest.update(str(type(v)).encode())
            for item in v:visit(item)
        else:digest.update(json.dumps(v,sort_keys=True,default=str,allow_nan=False).encode())
        digest.update(b']')
    visit(value);return digest.hexdigest()
