import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
from scipy.special import binom as _binom

class  hidalgo():

	def __init__(self, metric = 'euclidean', K = 1, zeta=0.8, q=3, Niter=10, Nreplicas=1,burn_in=0.5,a=np.ones(2),b=np.ones(2),c=np.ones(2),f=np.ones(2)):

		a=np.ones(K)
		b=np.ones(K)
		c=np.ones(K)
		f=np.ones(K)

		self.metric = metric
		self.K = K
		self.zeta = zeta
		self.q = q
		self.Niter=Niter
		self.burn_in=burn_in
		self.Nreplicas=Nreplicas
		self.a=a
		self.b=b
		self.c=c
		self.f=f

	def _fit(self,X):

		q=self.q
		K=self.K
		Niter=self.Niter
		zeta=self.zeta
		Nreplicas=self.Nreplicas
		a=self.a
		b=self.b
		c=self.c
		f=self.f
		burn_in=self.burn_in

		assert isinstance(X,np.ndarray), "X should be a numpy array"
		assert len(np.shape(X))==2, "X should be a two-dimensional numpy array"

		N,d = np.shape(X)

		if self.metric!='predefined':
			nbrs = NearestNeighbors(n_neighbors=q+1, algorithm='ball_tree',metric=self.metric).fit(X)
			distances, indicesIn = nbrs.kneighbors(X)

			nbrmat=np.zeros((N,N))

			for n in range(q):
				nbrmat[indicesIn[:,0],indicesIn[:,n+1]]=1

			nbrcount=np.sum(nbrmat,axis=0)
			indicesOut=np.where(nbrmat.T)[1]
			indicesTrack=np.cumsum(nbrcount)
			indicesTrack=np.append(0,indicesTrack[:-1])
			
		else:
			distances = np.sort(X)[:,:q+1]
			indicesIn = np.argsort(X)[:,:q+1]

			nbrmat=np.zeros((N,N))

			for n in range(q):
				nbrmat[indicesIn[:,0],indicesIn[:,n+1]]=1

			nbrcount=np.sum(nbrmat,axis=0)
			indicesOut=np.where(nbrmat.T)[1]
			indicesTrack=np.cumsum(nbrcount)
			indicesTrack=np.append(0,indicesTrack[:-1])

		mu = np.divide(distances[:,2],distances[:,1])

		fixed_Z=1;
		use_Potts=1;
		estimate_zeta=0;
		sampling_rate=2;
		Nsamp=np.floor((Niter-np.ceil(burn_in*Niter))/sampling_rate).astype(int)
		self.Nsamp=Nsamp
		Npar=N+2*K+2+1*(estimate_zeta);

		sampling=2*np.ones(Nsamp*Npar);
		bestsampling=np.zeros((Nsamp,Npar));

		indicesIn=indicesIn[:,1:]
		indicesIn=np.reshape(indicesIn,(N*q,))

		maxlik=-1.E10

		r=1
		return (Niter,K,fixed_Z,use_Potts,estimate_zeta,q,zeta,sampling_rate,burn_in,r,mu,indicesIn.astype(float),indicesOut.astype(float),nbrcount,indicesTrack,a,b,c,f,sampling)
			

		for r in range(Nreplicas):
			print(Niter,K,fixed_Z,use_Potts,estimate_zeta,q,zeta,sampling_rate,burn_in,r,mu,indicesIn.astype(float),indicesOut.astype(float),nbrcount,indicesTrack,a,b,c,f,sampling)
			sampling=np.reshape(sampling,(Nsamp,Npar))	
			lik=np.mean(sampling[:,-1],axis=0)
			if(lik>maxlik): 
				bestsampling=sampling
			sampling=np.reshape(sampling,(Nsamp*Npar,))	

		return bestsampling



	def fit(self, X):
		N=np.shape(X)[0]
		
		return self._fit(X)
		
		sampling=self._fit(X)
		K=self.K

		self.d_=np.mean(sampling[:,:K],axis=0)
		self.derr_=np.std(sampling[:,:K],axis=0)
		self.p_=np.mean(sampling[:,K:2*K],axis=0)
		self.perr_=np.std(sampling[:,K:2*K],axis=0)
		self.lik_=np.mean(sampling[:,-1],axis=0)
		self.likerr_=np.std(sampling[:,-1],axis=0)
		
		Pi=np.zeros((K,N))

		for k in range(K):
			Pi[k,:]=np.sum(sampling[:,2*K:2*K+N]==k,axis=0)

		self.Pi=Pi/self.Nsamp
		Z=np.argmax(Pi,axis=0);
		pZ=np.max(Pi,axis=0);
		Z=Z+1
		Z[np.where(pZ<0.8)]=0
		self.Z=Z

def binom(N, q):
    ss=1.
    if q==0: return 1.
    for q1 in range(q):
        ss=ss*(N-q1)/(q1+1)
    return ss

    #return _binom(N,q)

# this has to be a bespoke function, because
#print(binom(-1,0)) -> 1
#print(binom(-1,1)) -> -1
#print(binom(-1,2)) -> 1
#print(binom(-1,3)) -> -1
# whereas in scipy.special, returns nan

# partition function Z
def Zpart(N, N1, zeta, q):
    s=0
    for q1 in range(q+1):
        s += binom(N1-1,q1) * binom(N-N1,q-q1) * zeta**(q1) * (1-zeta)**(q-q1)
    
    return s


#################### scenario ######################

# get model
K=2

# generate dataset
N=5
d1=1
d2=3

np.random.seed(10002)
X=np.zeros((2*N,6))

for j in range(d1):
	X[:N,j]= np.random.normal(0,3,N)

for j in range(d2):
	X[N:,j]= np.random.normal(2,1,N)


model=hidalgo(K=K)
Niter,K,fixed_Z,use_Potts,estimate_zeta,q,zeta,sampling_rate,burn_in,r,mu,indicesIn,indicesOut,nbrcount,indicesTrack,a,b,c,f,sampling = model.fit(X)

# name conflicts
N_ = mu.size
MU = mu
Iin = indicesIn.astype(int)
Iout = indicesOut.astype(int)
Iout_count = nbrcount.astype(int)
Iout_track = indicesTrack.astype(int)


print('int Niter = ',Niter,';', sep='')
print('const int K = ',K,';', sep='')
print('bool fixed_Z = ',fixed_Z,';', sep='')
print('bool use_Potts = ',use_Potts,';', sep='')
print('bool estimate_zeta = ',estimate_zeta,';', sep='')
print('int q = ',q,';', sep='')
print('vector <double> MU = {', ', '.join(mu.astype(str)),'};', sep='')
print('vector<int> Iin = {',', '.join(indicesIn.astype(int).astype(str)),'};',sep='')
print('vector<int> Iout = {',', '.join(indicesOut.astype(int).astype(str)),'};',sep='')
print('int Iout_count[] = {', ', '.join(nbrcount.astype(int).astype(str)),'};', sep='')
print('int Iout_track[] = {', ', '.join(indicesTrack.astype(int).astype(str)),'};', sep='')
print('double a[] = {',', '.join(a.astype(str)),'};', sep='')
print('double b[] = {',', '.join(b.astype(str)),'};', sep='')
print('double c[] = {',', '.join(c.astype(str)),'};', sep='')
print('double f[] = {',', '.join(f.astype(str)),'};', sep='')
print('double zeta = ',zeta,';', sep='')
print('vector <double> sampling = {',', '.join(sampling.astype(str)),'};', sep='')
print('int sampling_rate = ',sampling_rate,';', sep='')
print('int replica = ',r,';', sep='')

# burn_in is hardcoded in c++ code
# replaced N=MU.size();
print('const int N = ',mu.size,';', sep='')


# paste the above into `gibbs_katie_print_random.cpp` and retrive
# the randomly generated numbers for run as `random_numbers.csv`

import pandas as pd
random_list = pd.read_csv('python/random_numbers.csv', header=None).values.tolist()[0]

#### INITIALIZE PARAMETERS ###############################

# params to initialise
V = np.empty(shape=K)
NN = np.empty(shape=K)
d = np.empty(shape=K)
p = np.empty(shape=K)
a1 = np.empty(shape=K)
b1= np.empty(shape=K)
c1 = np.empty(shape=K)
Z = np.empty(shape=N_, dtype=int)
f1 = np.empty(shape=2)

for k in range(K):
    V[k]=0 
    NN[k]=0
    d[k]=1
    sampling = np.append(sampling, d[k])

for k in range(K):
    p[k]=1./K
    sampling = np.append(sampling, p[k])

pp=(K-1)/K

for i in range(N_):
    z = random_list.pop(0)
    if fixed_Z==False:
        #z = int(np.floor(random.random()*K))
        Z[i]=z
    else:
        Z[i]=0

    V[Z[i]]=V[Z[i]]+np.log(MU[i])
    NN[Z[i]]+=1
    sampling = np.append(sampling, Z[i])

for k in range(K):
    a1[k]=a[k]+NN[k]
    b1[k]=b[k]+V[k]
    c1[k]=c[k]+NN[k]

N_in=0

for i in range(N_):
    k=Z[i]

    for j in range(q):
        index = Iin[q*i+j]
        if Z[index]==k: N_in +=1


f1[0]=f[0]+N_in; 
f1[1]=f[1]+N*q-N_in

sampling = np.append(sampling, 0)
sampling = np.append(sampling, 0)

########## check scenario 1 ###########

Z_check = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]
V_check = [6.35105, 0, ]
a1_check = [11.,1.,]
b1_check = [7.35105, 1, ]
MU_check = [1.46472, 5.40974, 2.46472, 4.78175, 1.17461, 1.95891, 1.05864, 1.06313, 1.85041, 1.28065, ]
NN_check = [10, 0, ]
N_check = 10
sampling_check = [
2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]
N_in_check = 30


assert(N_ == N_check)
assert(N_in == N_in_check)
assert all(NN == NN_check)
assert all((math.isclose(n1, n2, abs_tol = 0.0002) for n1, n2 in zip(MU, MU_check)))
assert all(Z == Z_check)
assert all((math.isclose(n1, n2, abs_tol = 0.0002) for n1, n2 in zip(V, V_check)))
assert all(sampling == sampling_check)

#########################################

for it in range(Niter):

    #### SAMPLING d ###############################
    for k in range(K):
        stop = False

        while stop ==False:

            #r1 = random.random()*200 # random sample for d[k]
            #r2 = random.random() # random number for accepting

            r1 = random_list.pop(0)
            r2 = random_list.pop(0)

            rmax = (a1[k]-1)/b1[k]

            if (a1[k]-1>0): 
                frac = np.exp(-b1[k]*(r1-rmax)-(a1[k]-1)*(np.log(rmax)-np.log(r1))) 
            else:
                frac = np.exp(-b1[k]*r1)

            if (frac>r2):
                stop=True
                if(it%sampling_rate==0 and it>= Niter*burn_in):
                    sampling = np.append(sampling, r1)
                d[k]=r1


    #### SAMPLING p ###############################
    for k in range(K-1):
        stop = False

        while stop ==False:

            #r1 = random.random() # random sample for p[k]
            #r2 = random.random() # random number for accepting

            r1 = random_list.pop(0)
            r2 = random_list.pop(0)

            rmax = (c1[k]-1)/(c1[k]-1+c1[K-1]-1)
            frac = ((r1/rmax)**(c1[k]-1))*(((1-r1)/(1-rmax))**(c1[K-1]-1))

            if(frac>r2):
                stop=True
                r1=r1*(1.-pp+p[k])
                p[K-1]+=p[k]-r1
                pp-=p[k]-r1
                p[k]=r1
                if(it%sampling_rate==0 and it>= Niter*burn_in):
                    sampling = np.append(sampling, r1)

    if(it%sampling_rate==0 and it>= Niter*burn_in):
        sampling = np.append(sampling, (1-pp))


    #### SAMPLING zeta ###############################
    stop=False
    maxval=-100000
    mx=0
    l=0

    if bool(use_Potts)==True and bool(estimate_zeta)==True:
        for l in range(10):
            zeta1=0.5+0.05*l
            ZZ = np.empty((K,0))
            for k in range(K):
                ZZ =np.append(ZZ, Zpart(N_, NN[k], zeta1, q))
            h = 0
            for k in range(K):
                h=h+NN[k]*np.log(ZZ[k])
            
            val = (f1[0]-1)*np.log(zeta1)+(f1[1]-1)*np.log(1-zeta1)-h

            if(val > maxval):
                maxval=val # found max val for below frac
                mx=zeta1

        while stop == False:
            #r1 = random.random() # random sample for zeta
            #r2 = random.random() # random number for accepting

            r1 = random_list.pop(0)
            r2 = random_list.pop(0)

            ZZ = np.empty((K,0))
            for k in range(K):
                ZZ =np.append(ZZ, Zpart(N_, NN[k], r1, q))
            h = 0
            for k in range(K):
                h=h+NN[k]*np.log(ZZ[k])
            
            val = (f1[0]-1)*np.log(r1)+(f1[1]-1)*np.log(1-r1)-h
            frac = np.exp(val - maxval)

            if frac > r2:
                stop = True
                if it > 0: zeta = r1

    #### SAMPLING Z ###############################

    for i in range(N_):

        if fixed_Z == True: break

        if abs((zeta-1)<1E-5):
            if(it%sampling_rate==0 and it>= Niter*burn_in):
                sampling = np.append(sampling, Z[i])
                continue
        
        stop = False
        prob = np.empty(shape=K)
        gg = np.empty(shape=K)
        norm = 0
        gmax = 0

        for k1 in range(K):
            g = 0
            if use_Potts == True:
                n_in = 0
                for j in range(q):
                    index = int(Iin[q*i+j])
                    if Z[index]==k1: n_in=n_in +1.
                m_in = 0
                for j in range(int(Iout_count[i])):
                    index = int(Iout[Iout_track[i] + j])
                    if index > -1 and Z[index] == k1: m_in = m_in + 1.

                g = (n_in+m_in)*np.log(zeta/(1-zeta))-np.log(Zpart(N_,NN[k1],zeta,q))
                g=g+np.log(Zpart(N_,NN[k1]-1,zeta,q)/Zpart(N_,NN[k1],zeta,q))*(NN[k1]-1)

            if g > gmax: gmax=g
            gg[k1]=g

        for k1 in range(K):
            gg[k1]=np.exp(gg[k1]-gmax)

        for k1 in range(K):
            prob[k1]=p[k1]*d[k1]*MU[i]**(-(d[k1]+1))*gg[k1]
            norm+=prob[k1]

        for k1 in range(K):
            prob[k1]=prob[k1]/norm
        
        while_loop_count = 0
        while stop == False:

            while_loop_count += 1
            if while_loop_count > 10000: break

            #r1 = int(np.floor(random.random()*K))
            #r2 = random.random()

            r1 = random_list.pop(0)
            r2 = random_list.pop(0)

            if prob[r1] > r2:
                stop = True
                if(it%sampling_rate==0 and it>= Niter*burn_in):
                    sampling = np.append(sampling, r1)
                NN[Z[i]]-=1
                a1[Z[i]]-=1
                c1[Z[i]]-=1
                V[Z[i]]-=np.log(MU[i])
                b1[Z[i]]-=np.log(MU[i])
                Z[i]=r1
                NN[Z[i]]+=1
                a1[Z[i]]+=1
                c1[Z[i]]+=1
                V[Z[i]]+=np.log(MU[i])
                b1[Z[i]]+=np.log(MU[i])

    #### updating prior on zeta ###############################

    N_in = 0
    for i in range(N_):
        k=Z[i]

        for j in range(q):
            index=int(Iin[q*i+j])

            if(Z[index]==k): N_in=N_in+1.0


    f1[0]=f[0]+N_in
    f1[1]=f[1]+N_*q-N_in

    #### likelihood ###############################
    lik0=0
    for i in range(N_):
        lik0=lik0+np.log(p[Z[i]])+np.log(d[Z[i]])-(d[Z[i]]+1)*np.log(MU[i])
    
    lik1=lik0+np.log(zeta/(1-zeta))*N_in

    for k1 in range(K):
        lik1=lik1-(NN[k1]*np.log(Zpart(N_, NN[k1], zeta, q)))

    if(it%sampling_rate==0 and it>= Niter*burn_in):
        print(it)
        sampling = np.append(sampling, lik0) 
        sampling = np.append(sampling, lik1)


# check all of the sampling==sampling from c++
# given the deterministic read of random vlaues
sampling_check = random_list
assert all((math.isclose(n1, n2, abs_tol = 0.02) for n1, n2 in zip(sampling, sampling_check)))
