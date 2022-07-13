import numpy as np
import pandas as pd
import math
from sklearn.neighbors import NearestNeighbors
from ..gibbs_katie_read_random import hidalgo

def _isclose(list1, list2):
    return all(math.isclose(n1, n2, abs_tol = 0.0002) for n1, n2 in zip(list1, list2))

# get model
K=2
model=hidalgo(K=K)

# generate dataset
N=10
N_half = int(N/2)
np.random.seed(10002)
X=np.zeros((N,6))

# half the points from one generating regime
for j in range(1):
	X[:N_half,j]= np.random.normal(0,3,N_half)

# the other half from another
for j in range(3):
	X[N_half:,j]= np.random.normal(2,1,N_half)

def test_X():
    assert isinstance(X,np.ndarray), "X should be a numpy array"
    assert len(np.shape(X))==2, "X should be a two-dimensional numpy array"

def test_get_neighbourhood_params():

    # get data related parameters
    MU,Iin,Iout,Iout_count,Iout_track = model._get_neighbourhood_params(X)

    MU_check = [1.46472, 5.40974, 2.46472, 4.78175, 1.17461, 1.95891, 1.05864, 1.06313, 1.85041, 1.28065, ]
    Iin_check = [2, 4, 7, 3, 9, 4, 0, 4, 7, 1, 4, 9, 0, 9, 3, 8, 6, 9, 8, 5, 9, 5, 8, 9, 5, 6, 7, 5, 7, 4]
    Iout_check = [2, 4, 3, 0, 1, 4, 0, 1, 2, 3, 9, 6, 7, 8, 9, 5, 8, 0, 2, 8, 9, 5, 6, 7, 1, 3, 4, 5, 6, 7]
    Iout_count_check = [2, 1, 1, 2, 5, 4, 2, 4, 3, 6]
    Iout_track_check = [0, 2, 3, 4, 6, 11, 15, 17, 21, 24]

    assert _isclose(MU, MU_check)
    assert _isclose(Iin, Iin_check)
    assert _isclose(Iout, Iout_check)
    assert _isclose(Iout_count, Iout_count_check)
    assert _isclose(Iout_track, Iout_track_check)


def test_initialise_params(model):

    Iin = [2, 4, 7, 3, 9, 4, 0, 4, 7, 1, 4, 9, 0, 9, 3, 8, 6, 9, 8, 5, 9, 5, 8, 9, 5, 6, 7, 5, 7, 4]
    
    # fix random numbers for testing
    random_list = [55.453,0.235881,91.2059,0.557685,152.582,0.218869,36.1272,0.713884,86.0666,0.210042]
    
    # initialise all other parameers, including randomly generated ones
    V, NN, d, p, a1, b1, c1, Z, f1, N_in, pp, sampling, _ = model._initialise_params(Iin, random_list)

    N_check = 10
    V_check = [6.35105, 0]
    NN_check = [10, 0]
    d_check = [1., 1.]
    p_check = [0.5, 0.5]
    a1_check = [11.,1.,]
    b1_check = [7.35105, 1, ]
    c1_check = [11.,  1.]
    Z_check = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]
    f1_check = [ 31., -14.]
    N_in_check = 30
    pp_check = 0.5
    sampling_check = [
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]

    assert model.N == N_check
    assert _isclose(V, V_check)
    assert _isclose(NN, NN_check)
    assert _isclose(d, d_check)
    assert _isclose(p, p_check)
    assert _isclose(a1, a1_check)
    assert _isclose(b1, b1_check)
    assert _isclose(c1, c1_check)
    assert _isclose(Z, Z_check)
    assert _isclose(f1, f1_check)
    assert N_in == N_in_check
    assert pp == pp_check
    assert _isclose(sampling, sampling_check)

def test_gibbs_sampling(model):
    
    MU = [1.46472, 5.40974, 2.46472, 4.78175, 1.17461, 1.95891, 1.05864, 1.06313, 1.85041, 1.28065, ]
    Iin = [2, 4, 7, 3, 9, 4, 0, 4, 7, 1, 4, 9, 0, 9, 3, 8, 6, 9, 8, 5, 9, 5, 8, 9, 5, 6, 7, 5, 7, 4]
    Iout = [2, 4, 3, 0, 1, 4, 0, 1, 2, 3, 9, 6, 7, 8, 9, 5, 8, 0, 2, 8, 9, 5, 6, 7, 1, 3, 4, 5, 6, 7]
    Iout_count = [2, 1, 1, 2, 5, 4, 2, 4, 3, 6]
    Iout_track = [0, 2, 3, 4, 6, 11, 15, 17, 21, 24]
    V= [6.35105, 0]
    NN = [10, 0]
    d = [1., 1.]
    p = [0.5, 0.5]
    a1 = [11.,1.,]
    b1 = [7.35105, 1, ]
    c1 = [11.,  1.]
    Z = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]
    f1 = [ 31., -14.]
    N_in = 30
    pp = 0.5
    sampling = [
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]

    sampling = model.gibbs_sampling(MU,Iin,Iout,Iout_count,Iout_track,V, NN, d, p, a1, b1, c1, Z, f1, N_in, pp, sampling)
    sampling_check = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,0.5,0.5,0,0,0,0,0,0,0,0,0,0,0,0,1.4907,0.34657,0.95813,0.0418701,-12.2538,-8.27885,1.73127,0.263247,0.879096,0.120904,-13.1465,-9.17152,]
    
    assert _isclose(sampling, sampling_check)