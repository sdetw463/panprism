import numpy as np
import pytest
from panprism.model_selection import fit_grouped_antisymmetric_ridge

def test_grouped_fit_matches_fixed_alpha_normal_equations_and_has_exact_parity():
    rng=np.random.default_rng(42);x=rng.normal(size=(90,5));y=rng.normal(size=90);g=np.repeat(['chr1','chr2','chr3'],30)
    fit=fit_grouped_antisymmetric_ridge(x,y,g,alphas=(10.,))
    rms=np.sqrt(np.mean(x*x,axis=0));z=x/rms
    expected=np.linalg.solve(z.T@z+10*np.eye(5),z.T@y)
    np.testing.assert_allclose(fit.weights,expected,rtol=1e-10,atol=1e-10)
    assert np.max(abs(fit.predict(x)+fit.predict(-x)))==0
    assert np.max(abs(fit.predict(np.zeros_like(x))))==0

def test_grouped_inner_selection_is_invariant_to_row_order():
    rng=np.random.default_rng(52);x=rng.normal(size=(120,4));y=x@np.array([.5,-.3,0.,.8])+rng.normal(size=120)*.4;g=np.repeat(['a','b','c','d'],30);mask=[True,False,True,True]
    a=fit_grouped_antisymmetric_ridge(x,y,g,mask);p=rng.permutation(len(y));b=fit_grouped_antisymmetric_ridge(x[p],y[p],g[p],mask)
    assert a.alpha==b.alpha
    np.testing.assert_allclose(a.predict(x),b.predict(x),atol=1e-9,rtol=1e-9)
    assert np.all(a.weights[np.array(mask)]>=0)

def test_single_group_does_not_silently_become_rowwise_cv():
    with pytest.raises(ValueError,match='two training groups'):
        fit_grouped_antisymmetric_ridge(np.ones((30,2)),np.ones(30),['one']*30)
