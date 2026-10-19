"""Small tested primitives for bounded, resumable score training."""
import torch

class BatchStream:
    def __init__(self,data,batch_size,seed,state=None):
        if len(data)<batch_size or batch_size<1:
            raise ValueError('Dataset must contain at least one complete batch')
        self.data=data
        self.batch_size=batch_size
        self.generator=torch.Generator().manual_seed(seed)
        self.permutation=torch.empty(0,dtype=torch.int64)
        self.cursor=0
        if state is not None:
            if state['size']!=len(data) or state['batch_size']!=batch_size:
                raise ValueError('Resume dataset size/batch differs')
            self.generator.set_state(state['rng'])
            self.permutation=state['permutation'].clone()
            self.cursor=int(state['cursor'])

    def next(self):
        if self.cursor+self.batch_size>len(self.permutation):
            self.permutation=torch.randperm(len(self.data),generator=self.generator)
            self.cursor=0
        index=self.permutation[self.cursor:self.cursor+self.batch_size]
        self.cursor+=self.batch_size
        return self.data[index]

    def state_dict(self):
        return {'size':len(self.data),'batch_size':self.batch_size,
                'rng':self.generator.get_state().clone(),
                'permutation':self.permutation.clone(),'cursor':self.cursor}

def freeze_teacher(model):
    model.eval()
    for param in model.parameters(): param.requires_grad_(False)

def optimizer_update(model,optimizer,loss_closure,step,learning_rate,warmup,grad_clip=1.):
    model.train()
    for group in optimizer.param_groups:
        group['lr']=learning_rate*min(step/warmup,1.) if warmup>0 else learning_rate
    optimizer.zero_grad(set_to_none=True)
    loss=loss_closure()
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip,error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach().item())
