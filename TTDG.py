import numpy as np
import scikit_tt.tensor_train as tt
from scipy.optimize import minimize

def left_right_interface(coreList, k, only_list=False):
    N = len(coreList)
    leftU = []
    rightV = []

    if k > 0:
        leftU = coreList[:k]
    if k < N-1:
        rightV = coreList[k+1:] 

    if only_list:
        return leftU, rightV  
    
    multU = np.ones((1,))
    for l in range(len(leftU)):
        multU = np.tensordot(multU, leftU[l], axes=([-1], [0]))
    
    multV= np.ones((1,))
    for r in range(len(rightV)-1, -1, -1): 
        multV = np.tensordot(rightV[r], multV, axes=([-1], [0]))

    return multU, multV


def loss_with_grad(T, Gcur):
    N = T.order
    grad_cores = []
    approx = tt.TT(Gcur)
    residual = approx-T
    loss = 0.5 * tt.TT.norm(residual)**2
    [kStart, kEnd, kStep] = [0, N, 1]

    for k in range(kStart, kEnd, kStep):
        leftU = []
        rightV = []
        if k > 0:
            leftU = Gcur[:k]
        if k < N-1:
            rightV = Gcur[k+1:] 
        Utt = tt.TT(leftU) if k > 0 else 1
        Vtt = tt.TT(rightV) if k < N-1 else 1
        
        UT = residual.tensordot(Utt, num_axes=k, mode='first-first') if k > 0 else residual
        UTV = UT.tensordot(Vtt, num_axes=N-(k+1), mode='last-last') if k < N-1 else UT
        grad_core_k = UTV.cores[0]
        grad_cores.append(grad_core_k)
    return loss, grad_cores

def pack(cores):
    """Flatten list of arrays to 1D vector."""
    return np.concatenate([c.ravel() for c in cores])

def unpack(x, template_cores):
    """Un-flatten 1D vector x into list of arrays with shapes like template_cores."""
    cores = []
    offset = 0
    for T in template_cores:
        size = np.prod(T.shape)
        cores.append(x[offset:offset+size].reshape(T.shape))
        offset += size
    return cores

def fun(x):
    Gcur = unpack(x, Ginit)
    loss, _ = loss_with_grad(T, Gcur)
    return loss

def jac(x):
    Gcur = unpack(x, Ginit)
    _, grads = loss_with_grad(T, Gcur)
    g = pack(grads)
    return g.astype(np.float64)


# -----------------------


Ginit = [
        np.random.randn(1, 5, 1, 3),
        np.random.randn(3, 6, 1, 2),
        np.random.randn(2, 7, 1, 2),
        np.random.randn(2, 8, 1, 3),
        np.random.randn(3, 4, 1, 1)
    ]

Go = [
        np.random.randn(1, 5, 1, 3),
        np.random.randn(3, 6, 1, 2),
        np.random.randn(2, 7, 1, 2),
        np.random.randn(2, 8, 1, 3),
        np.random.randn(3, 4, 1, 1)
    ]
N = len(Go)

T = tt.TT(Go)



x0 = pack(Ginit)
res = minimize(
    fun,        
    x0,          
    method='L-BFGS-B',
    jac=jac,     
    options={
        'maxiter': 1000,
        'gtol': 1e-10,
        'ftol': 1e-10,
        'disp': True
    }
)

Gopt = unpack(res.x, Ginit)
print("Final loss:", res.fun)

tt.TT.norm(tt.TT(Gopt)-T)/tt.TT.norm(T)

