"""Original full-object FP32 gates plus integer IEEE ULP descriptors."""
import torch
from common import metric,compare_tree
def ulp(a,r):
    if a is None or r is None:return {'same_activity':a is r}
    assert a.shape==r.shape and a.dtype==r.dtype
    if not a.is_floating_point():return {'bitwise_equal':torch.equal(a,r)}
    assert torch.isfinite(a).all() and torch.isfinite(r).all()
    if a.dtype==torch.float32:
        ai=a.contiguous().view(torch.int32).to(torch.int64);ri=r.contiguous().view(torch.int32).to(torch.int64)
        ao=torch.where(ai<0,-(ai&0x7fffffff),ai);ro=torch.where(ri<0,-(ri&0x7fffffff),ri)
        distance=(ao-ro).abs()
        return {'elements':a.numel(),'max_ulp':int(distance.max()),'equal_elements':int((distance==0).sum()),
            'within_one_ulp_elements':int((distance<=1).sum()),'value_dtype':'float32'}
    assert a.dtype==torch.float64
    # Energy outputs are small; Python integers avoid signed-int64 distance overflow.
    def ordered(v):return -(v&0x7fffffffffffffff) if v<0 else v
    distance=[abs(ordered(x)-ordered(y)) for x,y in zip(a.contiguous().view(torch.int64).flatten().tolist(),r.contiguous().view(torch.int64).flatten().tolist())]
    return {'elements':len(distance),'max_ulp':max(distance,default=0),'equal_elements':sum(v==0 for v in distance),
        'within_one_ulp_elements':sum(v<=1 for v in distance),'value_dtype':'float64'}
def full(a,r,prefix=''):
    result=compare_tree(a,r,prefix)
    def visit(x,y,path):
        if isinstance(y,dict):
            for k in y:visit(x[k],y[k],path+'/'+str(k))
        elif isinstance(y,(list,tuple)):
            for i,(xx,yy) in enumerate(zip(x,y)):visit(xx,yy,path+'/'+str(i))
        elif isinstance(y,torch.Tensor) or y is None:result[path]['ulp']=ulp(x,y)
    visit(a,r,prefix)
    return result
