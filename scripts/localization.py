"""Batched equivalents of the historical grid and normalized-gradient refinement."""
import numpy as np
import torch

class ArrayGeometry:
    def __init__(self,nx=64,ny=16,device='cpu',dtype=torch.float32):
        self.nx=nx; self.ny=ny; self.n=nx*ny; self.device=device; self.dtype=dtype
        self.x=torch.arange(nx,device=device,dtype=dtype)
        self.y=torch.arange(ny,device=device,dtype=dtype)
        self.ix=self.x[None,:].expand(ny,nx).reshape(-1)
        self.iy=self.y[:,None].expand(ny,nx).reshape(-1)
        self.bs=torch.tensor([0.,0.,25.],device=device,dtype=dtype)
        self.gx=torch.linspace(-1,1,2*nx,device=device,dtype=dtype)
        self.gy=torch.linspace(-1,1,2*ny,device=device,dtype=dtype)
        self.ax=torch.exp(1j*torch.pi*self.gx[:,None]*self.x)
        self.ay=torch.exp(1j*torch.pi*self.gy[:,None]*self.y)

    def mu_from_position(self,position):
        direction=position-self.bs
        direction=direction/torch.linalg.vector_norm(direction,dim=-1,keepdim=True)
        return direction[:,[0,2]]

    def position_from_mu(self,mu,clamp=True):
        remaining=1-(mu**2).sum(dim=-1)
        if clamp: remaining=remaining.clamp(min=0)
        direction=torch.stack([mu[:,0],remaining.sqrt(),mu[:,1]],dim=-1)
        direction=direction/torch.linalg.vector_norm(direction,dim=-1,keepdim=True)
        return self.bs+(-23.5/direction[:,2])[:,None]*direction

    def steering(self,mu,amplitude=None):
        ax=torch.exp(1j*torch.pi*mu[:,0,None]*self.x)
        ay=torch.exp(1j*torch.pi*mu[:,1,None]*self.y)
        result=ay[:,:,None]*ax[:,None,:]
        return result if amplitude is None else result*amplitude[:,None,None]

    def initialize(self,y,mode='2d'):
        px=y@self.ax.conj().T
        py=self.ay.conj()@y
        if mode=='1d':
            ix=px.abs().square().sum(dim=1).argmax(dim=1)
            iy=py.abs().square().sum(dim=2).argmax(dim=1)
            return self.position_from_mu(torch.stack([self.gx[ix],self.gy[iy]],dim=-1),clamp=False)
        if mode!='2d': raise ValueError('Unknown initialization')
        metric=(self.ay.conj()@px).abs().square().transpose(1,2)
        valid=self.gx[:,None]**2+self.gy[None,:]**2<1
        metric=metric.masked_fill(~valid[None],-torch.inf)
        index=metric.reshape(len(y),-1).argmax(dim=1)
        mu=torch.stack([self.gx[index//len(self.gy)],self.gy[index%len(self.gy)]],dim=-1)
        return self.position_from_mu(mu,clamp=False)

    def objective_gradient(self,y,mu,variance,precision=None):
        a=self.steering(mu).reshape(len(y),-1)
        y=y.reshape(len(y),-1)
        variance=variance.to(device=y.device,dtype=self.dtype)
        if precision is None:
            qy=y/variance[:,None]; qa=a/variance[:,None]
        else:
            qy=y@precision.T; qa=a@precision.T
        c=(a.conj()*qy).sum(dim=1)
        denominator=(a.conj()*qa).sum(dim=1).real
        power=c.abs().square()
        jacobian=torch.stack([1j*torch.pi*self.ix*a,1j*torch.pi*self.iy*a],dim=-1)
        dc=(qy.conj()[:,:,None]*jacobian).sum(dim=1)
        db=(qa.conj()[:,:,None]*jacobian).sum(dim=1)
        gradient=2*(c[:,None]*dc/denominator[:,None]-power[:,None]*db/denominator[:,None]**2).real
        return power/denominator,gradient

    def refine(self,y,initial_position,variance,precision=None,max_iters=100,tol=1e-4):
        mu=self.mu_from_position(initial_position).clone()
        active=torch.isfinite(mu).all(dim=1)
        lr=.01
        for iteration in range(max_iters):
            _,gradient=self.objective_gradient(y,mu,variance,precision)
            norm=torch.linalg.vector_norm(gradient,dim=-1)
            active=active & (norm>1e-10) & torch.isfinite(norm)
            update=lr*gradient/norm.clamp(min=1e-30)[:,None]
            candidate=mu+update
            length=torch.linalg.vector_norm(candidate,dim=-1)
            candidate=torch.where((length>1)[:,None],candidate*(.99/length)[:,None],candidate)
            difference=torch.linalg.vector_norm(candidate-mu,dim=-1)
            mu=torch.where(active[:,None],candidate,mu)
            active=active & (difference>=tol)
            if not bool(active.any()): break
            if (iteration+1)%10==0: lr*=.5
        a=self.steering(mu).reshape(len(y),-1); vector=y.reshape(len(y),-1)
        if precision is None:
            amplitude=(a.conj()*vector).sum(dim=1)/self.n
        else:
            qy=vector@precision.T; qa=a@precision.T
            amplitude=(a.conj()*qy).sum(dim=1)/(a.conj()*qa).sum(dim=1)
        return self.position_from_mu(mu),amplitude

def position_crb(position,noise_variance,los_power,nx=64,ny=16):
    position=np.asarray(position,dtype=float)
    x=position[...,0]; y=position[...,1]; dz=position[...,2]-25
    r=np.sqrt(x*x+y*y+dz*dz)
    jacobian=np.stack([np.stack([(r*r-x*x)/r**3,-x*y/r**3],axis=-1),
                       np.stack([-dz*x/r**3,-dz*y/r**3],axis=-1)],axis=-2)
    information=2*los_power*np.pi**2/noise_variance*np.diag([ny*nx*(nx**2-1)/12,nx*ny*(ny**2-1)/12])
    fim=np.swapaxes(jacobian,-1,-2)@information@jacobian
    return np.trace(np.linalg.inv(fim),axis1=-2,axis2=-1)
