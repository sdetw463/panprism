"""Ridge selection with group-disjoint inner folds and fold-specific RMS scaling."""
from dataclasses import dataclass
from typing import Dict,Sequence
import numpy as np
from scipy.optimize import lsq_linear

@dataclass
class GroupedRidgeFit:
    weights: np.ndarray
    rms: np.ndarray
    alpha: float
    inner_mse: Dict[str,float]

    def predict(self,x):
        return (np.asarray(x,dtype=np.float64)/self.rms)@self.weights

def _fit_fixed_alpha(x,y,alpha,nonneg_mask):
    rms=np.sqrt(np.mean(x*x,axis=0));rms=np.where(rms<1e-8,1.,rms)
    z=x/rms;q=z.T@z+alpha*np.eye(z.shape[1]);rhs=z.T@y
    lower=np.where(nonneg_mask,0.,-np.inf);chol=np.linalg.cholesky(q)
    fit=lsq_linear(chol.T,np.linalg.solve(chol,rhs),bounds=(lower,np.full(z.shape[1],np.inf)),tol=1e-11,max_iter=1000)
    if not fit.success:raise RuntimeError(f'Constrained ridge failed: {fit.message}')
    return fit.x,rms

def fit_grouped_antisymmetric_ridge(x,y,groups,nonneg_mask=None,alphas:Sequence[float]=(.5,10.,250.,4000.),inner_folds=3):
    """Fit only the provided training data; no intercept or target centering.

    Each group belongs to one inner validation fold. Each inner training fold
    fits its own RMS scales. Alpha minimizes pooled inner validation MSE.
    Outer splitting is the caller's responsibility.
    """
    x=np.asarray(x,np.float64);y=np.asarray(y,np.float64);groups=np.asarray(groups,str)
    if x.ndim!=2 or y.shape!=(len(x),) or groups.shape!=(len(x),):raise ValueError('Incompatible x, y or groups shapes')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite training data')
    unique=sorted(set(groups));nfold=min(inner_folds,len(unique))
    if nfold<2:raise ValueError('At least two training groups are required')
    if not alphas or min(alphas)<=0:raise ValueError('Positive ridge penalties are required')
    mask=np.zeros(x.shape[1],bool) if nonneg_mask is None else np.asarray(nonneg_mask,bool)
    if mask.shape!=(x.shape[1],):raise ValueError('nonneg_mask must match feature count')
    assignment={c:i%nfold for i,c in enumerate(unique)};fold=np.array([assignment[c] for c in groups]);loss={}
    for alpha in alphas:
        p=np.empty(len(y))
        for k in range(nfold):
            va=fold==k;tr=~va;w,r=_fit_fixed_alpha(x[tr],y[tr],alpha,mask);p[va]=(x[va]/r)@w
        loss[str(float(alpha))]=float(np.mean((p-y)**2))
    best=float(alphas[int(np.argmin(list(loss.values())))]);w,r=_fit_fixed_alpha(x,y,best,mask)
    return GroupedRidgeFit(w,r,best,loss)
