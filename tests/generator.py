"""
This file generates all the persistent containers needed to execute the random tests.
To run it, you need to have the iarray library (python wrapper).
"""


import iarray as ia
import numpy as np


def create_urlpath(method, dtype, **kwargs):
    # Create urlpath
    funcname = method.__name__
    dname = str(np.dtype(dtype))
    urlpath = f"test_{funcname}_{dname}"
    for k, v in kwargs.items():
        if k != "size":
            urlpath += f"_{k}{v}"
    urlpath += f".iarray"
    return urlpath


def create_files(method, dtype, **kwargs):
    shape = (1000 * 1000,)
    size = int(np.prod(shape))
    chunkshape = (100 * 1000,)
    blockshape = (10 * 1000,)

    with ia.config(clevel=5, codec=ia.Codecs.LZ4) as cfg:
        if kwargs:
            kwargs["size"] = size
            c = dtype(method(**kwargs))
        else:
            c = dtype(method(size))

        storage = ia.Storage(chunkshape, blockshape, create_urlpath(method, dtype, **kwargs))

        ia.numpy2iarray(c, storage=storage, cfg=cfg)


# Rand
create_files(np.random.rand, np.float64)
create_files(np.random.rand, np.float32)

# Randn
create_files(np.random.randn, np.float64)
create_files(np.random.randn, np.float32)

# Beta
create_files(np.random.beta, np.float64, a=3, b=4)
create_files(np.random.beta, np.float32, a=0.1, b=5)

# Lognormal
create_files(np.random.lognormal, np.float64, mean=3, sigma=4)
create_files(np.random.lognormal, np.float32, mean=0.1, sigma=5)

# Exponential
create_files(np.random.exponential, np.float64, scale=3)
create_files(np.random.exponential, np.float32, scale=0.1)

# Uniform
create_files(np.random.uniform, np.float64, low=-3, high=5)
create_files(np.random.uniform, np.float32, low=-0.1, high=0.2)

# Normal
create_files(np.random.normal, np.float64, loc=3, scale=5)
create_files(np.random.normal, np.float32, loc=0.1, scale=0.2)

# Binomial
create_files(np.random.binomial, np.float64, n=3, p=0.7)
create_files(np.random.binomial, np.float32, n=10, p=0.01)

# Poisson
create_files(np.random.poisson, np.float64, lam=3)
create_files(np.random.poisson, np.float32, lam=0.001)

# Bernoulli
create_files(np.random.binomial, np.float64, n=1, p=0.7)
create_files(np.random.binomial, np.float32, n=1, p=0.01)
