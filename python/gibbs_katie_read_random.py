import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
import pandas as pd
random_list = pd.read_csv('./random_numbers.csv', header=None).values.tolist()[0]


def binom(N, q):
    # this has to be a bespoke function, because
    #print(binom(-1,0)) -> 1
    #print(binom(-1,1)) -> -1
    #print(binom(-1,2)) -> 1
    #print(binom(-1,3)) -> -1
    # whereas in scipy.special, returns nan
    ss=1.
    if q==0: return 1.
    for q1 in range(q):
        ss=ss*(N-q1)/(q1+1)
    return ss

# partition function Z
def Zpart(N, N1, zeta, q):
    s=0
    for q1 in range(q+1):
        s += binom(N1-1,q1) * binom(N-N1,q-q1) * zeta**(q1) * (1-zeta)**(q-q1)
    
    return s
class hidalgo():

    def __init__(self,
        metric = 'euclidean', 
        K = 1,
        zeta=0.8,
        q=3,
        Niter=10,
        Nreplicas=1,
        burn_in=0.5,
        fixed_Z=1,
		use_Potts=1,
		estimate_zeta=0,
		sampling_rate=2,
        maxlik=-1.E10,
        a=None,
        b=None,
        c=None,
        f=None):

        if a is None: a=np.ones(K)
        if b is None: b=np.ones(K)
        if c is None: c=np.ones(K)
        if f is None: f=np.ones(K)

        self.metric = metric
        self.K = K
        self.zeta = zeta
        self.q = q
        self.Niter=Niter
        self.burn_in=burn_in
        self.Nreplicas=Nreplicas
        self.fixed_Z=fixed_Z
        self.use_Potts=use_Potts
        self.estimate_zeta=estimate_zeta
        self.sampling_rate=sampling_rate
        self.maxlik=maxlik
        self.a=a
        self.b=b
        self.c=c
        self.f=f

    def _get_neighbourhood_params(self, X):
        self.N,_ = np.shape(X)

        distances = np.sort(X)[:,:self.q+1]
        mu = np.divide(distances[:,2],distances[:,1])

        indicesIn = np.argsort(X)[:,:self.q+1]
        nbrmat=np.zeros((self.N,self.N))
        for n in range(self.q):
            nbrmat[indicesIn[:,0],indicesIn[:,n+1]]=1

        nbrcount=np.sum(nbrmat,axis=0).astype(int)
        indicesOut=np.where(nbrmat.T)[1].astype(int)
        indicesTrack=np.cumsum(nbrcount)
        indicesTrack=np.append(0,indicesTrack[:-1]).astype(int)
        indicesIn=indicesIn[:,1:]
        indicesIn=np.reshape(indicesIn,(self.N*self.q,)).astype(int)

        return (mu,indicesIn,indicesOut,nbrcount,indicesTrack)


    def _initialise_params(self, Iin, random_list):
        
        # params to initialise
        V = np.empty(shape=self.K)
        NN = np.empty(shape=self.K)
        d = np.empty(shape=self.K)
        p = np.empty(shape=self.K)
        a1 = np.empty(shape=self.K)
        b1= np.empty(shape=self.K)
        c1 = np.empty(shape=self.K)
        Z = np.empty(shape=self.N, dtype=int)
        f1 = np.empty(shape=2)

        for k in range(self.K):
            V[k]=0 
            NN[k]=0
            d[k]=1
            sampling = np.append(sampling, d[k])

        for k in range(self.K):
            p[k]=1./self.K
            sampling = np.append(sampling, p[k])

        pp=(self.K-1)/self.K

        for i in range(self.N):
            z = random_list.pop(0)
            if self.fixed_Z==False:
                #z = int(np.floor(random.random()*K))
                Z[i]=z
            else:
                Z[i]=0

            V[Z[i]]=V[Z[i]]+np.log(MU[i])
            NN[Z[i]]+=1
            sampling = np.append(sampling, Z[i])

        for k in range(self.K):
            a1[k]=self.a[k]+NN[k]
            b1[k]=self.b[k]+V[k]
            c1[k]=self.c[k]+NN[k]

        N_in=0

        for i in range(self.N):
            k=Z[i]

            for j in range(self.q):
                index = Iin[self.q*i+j]
                if Z[index]==k: N_in +=1


        f1[0]=self.f[0]+N_in; 
        f1[1]=self.f[1]+self.N*self.q-N_in

        sampling = np.append(sampling, 0)
        sampling = np.append(sampling, 0)

        return (V, NN, d, p, a1, b1, c1, Z, f1, N_in, pp, sampling, random_list)

    def gibbs_sampling(self, MU,Iin,Iout,Iout_count,Iout_track,V, NN, d, p, a1, b1, c1, Z, f1, N_in, pp, sampling):
        N_ = self.N
        
        for it in range(self.Niter):

            #### SAMPLING d ###############################
            for k in range(self.K):
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
                        if(it%self.sampling_rate==0 and it>= self.Niter*self.burn_in):
                            sampling = np.append(sampling, r1)
                        d[k]=r1


            #### SAMPLING p ###############################
            for k in range(self.K-1):
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
                        if(it%self.sampling_rate==0 and it>= self.Niter*self.burn_in):
                            sampling = np.append(sampling, r1)

            if(it%self.sampling_rate==0 and it>= self.Niter*self.burn_in):
                sampling = np.append(sampling, (1-pp))


            #### SAMPLING zeta ###############################
            stop=False
            maxval=-100000
            mx=0
            l=0

            if bool(self.use_Potts)==True and bool(self.estimate_zeta)==True:
                for l in range(10):
                    zeta1=0.5+0.05*l
                    ZZ = np.empty((self.K,0))
                    for k in range(self.K):
                        ZZ =np.append(ZZ, Zpart(N_, NN[k], zeta1, self.q))
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

                    ZZ = np.empty((self.K,0))
                    for k in range(self.K):
                        ZZ =np.append(ZZ, Zpart(N_, NN[k], r1, self.q))
                    h = 0
                    for k in range(self.K):
                        h=h+NN[k]*np.log(ZZ[k])
                    
                    val = (f1[0]-1)*np.log(r1)+(f1[1]-1)*np.log(1-r1)-h
                    frac = np.exp(val - maxval)

                    if frac > r2:
                        stop = True
                        if it > 0: zeta = r1

            #### SAMPLING Z ###############################

            for i in range(N_):

                if self.fixed_Z == True: break

                if abs((zeta-1)<1E-5):
                    if(it%self.sampling_rate==0 and it>= self.Niter*self.burn_in):
                        sampling = np.append(sampling, Z[i])
                        continue
                
                stop = False
                prob = np.empty(shape=K)
                gg = np.empty(shape=K)
                norm = 0
                gmax = 0

                for k1 in range(K):
                    g = 0
                    if self.use_Potts == True:
                        n_in = 0
                        for j in range(q):
                            index = int(Iin[q*i+j])
                            if Z[index]==k1: n_in=n_in +1.
                        m_in = 0
                        for j in range(int(Iout_count[i])):
                            index = int(Iout[Iout_track[i] + j])
                            if index > -1 and Z[index] == k1: m_in = m_in + 1.

                        g = (n_in+m_in)*np.log(zeta/(1-zeta))-np.log(Zpart(N_,NN[k1],zeta,self.q))
                        g=g+np.log(Zpart(N_,NN[k1]-1,zeta,self.q)/Zpart(N_,NN[k1],zeta,self.q))*(NN[k1]-1)

                    if g > gmax: gmax=g
                    gg[k1]=g

                for k1 in range(self.K):
                    gg[k1]=np.exp(gg[k1]-gmax)

                for k1 in range(self.K):
                    prob[k1]=p[k1]*d[k1]*MU[i]**(-(d[k1]+1))*gg[k1]
                    norm+=prob[k1]

                for k1 in range(self.K):
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
                        if(it%self.sampling_rate==0 and it>= self.Niter*self.burn_in):
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

                for j in range(self.q):
                    index=int(Iin[self.q*i+j])

                    if(Z[index]==k): N_in=N_in+1.0


            f1[0]=self.f[0]+N_in
            f1[1]=self.f[1]+N_*self.q-N_in

            #### likelihood ###############################
            lik0=0
            for i in range(N_):
                lik0=lik0+np.log(p[Z[i]])+np.log(d[Z[i]])-(d[Z[i]]+1)*np.log(MU[i])
            
            lik1=lik0+np.log(zeta/(1-zeta))*N_in

            for k1 in range(self.K):
                lik1=lik1-(NN[k1]*np.log(Zpart(N_, NN[k1], zeta, self.q)))

            if(it%self.sampling_rate==0 and it>= self.Niter*self.burn_in):
                print(it)
                sampling = np.append(sampling, lik0) 
                sampling = np.append(sampling, lik1)
        
        return sampling

    def _fit(self, X, random_list):

        MU,Iin,Iout,Iout_count,Iout_track = self._get_neighbourhood_params(X)
        V, NN, d, p, a1, b1, c1, Z, f1, N_in, pp, sampling = self._initialise_params(Iin, random_list)

        Nsamp=np.floor((self.Niter-np.ceil(self.burn_in*self.Niter))/self.sampling_rate).astype(int)
        Npar=self.N+2*self.K+2+1*(self.estimate_zeta)

        sampling=2*np.ones(Nsamp*Npar)
        bestsampling=np.zeros((Nsamp,Npar))

        # iterate through Nreplicas random starts and get posterior
        # samples with best max likelihood
        
        ## todo: parallel
        for r in range(self.Nreplicas):
            sampling = self.gibbs_sampling(MU,Iin,Iout,Iout_count,Iout_track,V, NN, d, p, a1, b1, c1, Z, f1, N_in, pp, sampling)
            sampling=np.reshape(sampling,(Nsamp,Npar))	
            lik=np.mean(sampling[:,-1],axis=0)
            if(lik>self.maxlik): 
                bestsampling=sampling
            sampling=np.reshape(sampling,(Nsamp*Npar,))	

        return bestsampling

    def fit(self, X):
        N=np.shape(X)[0]
        
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
        Z=np.argmax(Pi,axis=0)
        pZ=np.max(Pi,axis=0)
        Z=Z+1
        Z[np.where(pZ<0.8)]=0
        self.Z=Z

