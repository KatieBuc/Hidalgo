#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include <complex>
#include <functional>
#include <algorithm>
#include <numeric>
#include <assert.h>

using namespace std;

// variables passed in from python via Gibbs_Potts

///////////////////////////// BEGIN PASTE /////////////////////////
int Niter = 10;
const int K = 2;
bool fixed_Z = 1;
bool use_Potts = 1;
bool estimate_zeta = 0;
int q = 3;
vector <double> MU = {1.4647219996062466, 5.409740134317853, 2.4647219996062466, 4.7817501050892925, 1.1746064081642558, 1.9589087445409834, 1.058637057660818, 1.0631338562261194, 1.8504063601074203, 1.2806526635220237};
vector<int> Iin = {2, 4, 7, 3, 9, 4, 0, 4, 7, 1, 4, 9, 0, 9, 3, 8, 6, 9, 8, 5, 9, 5, 8, 9, 5, 6, 7, 5, 7, 4};
vector<int> Iout = {2, 4, 3, 0, 1, 4, 0, 1, 2, 3, 9, 6, 7, 8, 9, 5, 8, 0, 2, 8, 9, 5, 6, 7, 1, 3, 4, 5, 6, 7};
int Iout_count[] = {2, 1, 1, 2, 5, 4, 2, 4, 3, 6};
int Iout_track[] = {0, 2, 3, 4, 6, 11, 15, 17, 21, 24};
double a[] = {1.0, 1.0};
double b[] = {1.0, 1.0};
double c[] = {1.0, 1.0};
double f[] = {1.0, 1.0};
double zeta = 0.8;
vector <double> sampling = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
int sampling_rate = 2;
int replica = 1;
const int N = 10;
///////////////////////////// END PASTE /////////////////////////

// Create and open a text file
 ofstream MyFile("random_numbers.txt");

// variables declared in function Gibbs_Potts, initialized at zero
double d[K];
double V[K];	
int NN[K];
int Z[N];
double a1[K];
double b1[K];
double c1[K];
double p[K];
double pp;
double f1[2];


// binomial coefficient
double binom(int N, int q) {

 double s=1;
 
 if(q==0) return s;
 
 else{ 
 	for(int q1=0; q1<q; q1++) {
              s=s*(N-q1)/(q1+1);
   	}
	return s; 
 }

}

// partition function Z
double Zpart(int N, int N1, double zeta, int q){

        double s=0;

        for(int q1=0; q1<=q; q1++) {
	     s=s+binom(N1-1,q1)*binom(N-N1,q-q1)*pow(zeta,q1)*pow(1-zeta,q-q1);	

        }
        return s;
}



int main()
{
    //for (auto i: sampling)
    //std::cout << i << ' ';
    //cout << endl;;
    
    /*************** INITIALIZE PARAMETERS *************************************************************************************/
    srand(10000*replica);

    for(int k=0; k<K; k++) { 
        V[k]=0;  
        NN[k]=0;
        d[k]=1;
        sampling.push_back(d[k]);
    }

	//cout <<	"p[k] is" ;
    for(int k=0; k<K; k++) {
        p[k]=1./K;
        sampling.push_back(p[k]);
		//cout << p[k] << " ";
    }

	//cout <<	"pp is" ;
    pp=(double(K-1))/K;
	//cout << pp << endl;

    //  if(use_Potts==true) sampling.push_back(zeta);


    for(int i=0; i<N; i++){
        if(fixed_Z==false) { 
            int z=rand()%K;
            MyFile << z << ",";
            Z[i]=z;
        }
        else Z[i]=0;
        //cout << Z[i] << endl;;

        V[Z[i]]=V[Z[i]]+log(MU[i]);
        NN[Z[i]]+=1;
        sampling.push_back(Z[i]);
    }

    for(int k=0; k<K; k++) {
        a1[k]=a[k]+NN[k];
        b1[k]=b[k]+V[k];
        c1[k]=c[k]+NN[k];
    }

    double N_in=0;
    for(int i=0; i< N; i++) {
        int k=Z[i];
        for(int j=0; j<q;j++) {
            int index=Iin.at(q*i+j);
            if(Z[index]==k) N_in=N_in++;
        }			
    }	


    f1[0]=f[0]+N_in; 
    f1[1]=f[1]+N*q-N_in;

        sampling.push_back(0);
        sampling.push_back(0);
    
	
    cout << "Z = np.array([";
	for (auto i: Z)
    std::cout << i << ", ";
    cout << "])" << endl;

	cout << "V = np.array([";
	for (auto i: V)
    std::cout << i << ", ";
    cout << "])" << endl;
	
	cout << "a1 = [";
	for (auto i: a1)
    std::cout << i << ".," ;
    cout << "]" << endl;

	cout << "b1 = [";
	for (auto i: b1)
    std::cout << i << ", ";
    cout << "]" << endl;

	cout << "c1 = [";
	for (auto i: c1)
    std::cout << i << ", ";
    cout << "]" << endl;
    
	cout << "d = [";
	for (auto i: d)
    std::cout << i << ", ";
    cout << "]" << endl;

	cout << "p = [";
	for (auto i: p)
    std::cout << i << ", ";
    cout << "]" << endl;

	cout << "NN = [";
	for (auto i: NN)
    std::cout << i << "., ";
    cout << "]" << endl;

	cout << "f1 = [";
	for (auto i: f1)
    std::cout << i << "., ";
    cout << "]" << endl;

	cout << "pp = " << pp << endl;
	cout << "par = K"  << endl;


	cout << "sampling_check = [" << endl;
	for (auto i: sampling)
    std::cout << i << ", ";
    cout << "]" << endl;

	cout << "Z_check = [" << endl;
	for (auto i: Z)
    std::cout << i << ", ";
    cout << "]" << endl;

	cout << "V_check = [" << endl;
	for (auto i: V)
    std::cout << i << ", ";
    cout << "]" << endl;

	cout << "MU_check = [" << endl;
	for (auto i: MU)
    std::cout << i << ", ";
    cout << "]" << endl;

	cout << "NN_check = [" << endl;
	for (auto i: NN)
    std::cout << i << ", ";
    cout << "]" << endl;
	
	cout << "N_check = " << N << endl;

/******************* ITERATIONS **********************************************************************************************************/
	
	for(int it=0; it<Niter; it++){

		//if(it%100==0) mexPrintf("it= %d \n",it);

		bool stop;


		/* SAMPLING d***********************************************************************************************************/

		for(int k=0; k<K; k++){
	        
			stop=false;

			while(stop==false){
	   
				double r1 = double(rand())/RAND_MAX*200;
				MyFile << r1 << ",";
				double r2 = double(rand())/RAND_MAX;
				MyFile << r2 << ",";

				double rmax = (a1[k]-1)/b1[k];
				double frac;

				if(a1[k]-1>0) frac = exp(-b1[k]*(r1-rmax)-(a1[k]-1)*(log(rmax)-log(r1))); 
				else frac = exp(-b1[k]*r1);
				
				if(it==0) cout << "frac" << frac << endl;

				if(frac>r2){  
					stop=true;
					if(it%sampling_rate==0 &&  it>= Niter/10*5){
					/*
					cout << "it " << it << " ";
					cout << "k " << k << " ";
					cout << "r1 " << r1 << " ";
					cout << "r2 " << r2 << " ";
					cout << "frac " << frac << endl;*/
					sampling.push_back(r1);
                    /* for (auto i: sampling)
                    std::cout << i << ' ';
                    cout << endl;
					*/
					} 
					d[k]=r1; 
				}
			}
		}


		/* SAMPLING p **********************************************************************************************************/


		for(int k=0; k<K-1; k++) {

			stop=false;

			while(stop==false){

				double r1 = double(rand())/RAND_MAX; // random sample for p[k]
				MyFile << r1 << ",";
				double r2 = double(rand())/RAND_MAX; // random number for accepting
				MyFile << r2 << ",";
	
				double rmax = (c1[k]-1)/(c1[k]-1+c1[K-1]-1);
				double frac = pow(r1/rmax,c1[k]-1)*pow((1-r1)/(1-rmax),c1[K-1]-1);

				if(frac>r2){
					stop=true;
					r1=r1*(1.-pp+p[k]);
					p[K-1]+=p[k]-r1;
					pp-=p[k]-r1;  
					p[k]=r1;
					if(it%sampling_rate==0  &&  it>= Niter/10*5){
					/*
					cout << "it " << it << " ";
					cout << "k " << k << " ";
					cout << "r1 " << r1 << " ";
					cout << "r2 " << r2 << " ";
					cout << "frac " << frac << endl;*/
					sampling.push_back(r1);
                    /*for (auto i: sampling)
                    std::cout << i << ' ';
                    cout << endl;
					*/
					} 
				}
            } 
		}                                                                                                                                               		
		if(it%sampling_rate==0  &&  it>= Niter/10*5) sampling.push_back(1-pp);



              /* SAMPLING zeta ********************************************************************************************************/


                stop=false;

		double maxval=-100000;		
		double mx=0;

		int l=0;		

		while(l<10 && use_Potts==true && estimate_zeta==true) {
		
			double zeta1=0.5+0.05*l;

			double ZZ[K];
                        for(int k=0; k< K; k++) ZZ[k]=Zpart(N, NN[k], zeta1, q); // bug here, should be Zpart(N, NN[k], zeta1, q) 

			double h=0;
                        for(int k=0; k< K; k++) h=h+NN[k]*log(ZZ[k]);

                        double val=(f1[0]-1)*log(zeta1)+(f1[1]-1)*log(1-zeta1)-h;
			/*
			if(it == 0){
			cout << "ZZ is ";
			for (auto i: ZZ)
			std::cout << i << ' ';
			cout << endl;
			cout << "it " << it << " ";
			cout << "l " << l << " ";
			cout << "val " << val << endl;
			}
			*/
			if(val > maxval) { 
				maxval=val;
				mx=zeta1;
			}
			l++;
		}
	
                while(stop==false && use_Potts==true && estimate_zeta==true){


                        double r1 = double(rand())/RAND_MAX; // random sample for zeta
                        MyFile << r1 << ",";
                        double r2 = double(rand())/RAND_MAX; // random number for accepting
                        MyFile << r2 << ",";

                        double ZZ[K];
                        for(int k=0; k< K; k++) ZZ[k]=Zpart(N, NN[k], r1, q); //changed order, bug
                        
			double h=0;
			for(int k=0; k< K; k++) h=h+NN[k]*log(ZZ[k]);

                        double val=(f1[0]-1)*log(r1)+(f1[1]-1)*log(1-r1)-h;


                        double frac = val - maxval;

 
			frac=exp(frac);

                        if(frac>r2){
                                stop=true;
                                if(it > 0) zeta = r1;
                                //if(it%sampling_rate==0) sampling.push_back(r1);
                        }

                }

		//if(use_Potts==true && it%sampling_rate==0  &&  it>= Niter/10*5) sampling.push_back(zeta);
 

		/****** SAMPLING Z *******************************************************************************************************/


/*
		if(it==0){
		cout << "****" <<endl;
    
		cout << "d = [";
		for (auto i: d)
		std::cout << i << ", ";
		cout << "]" << endl;

		cout << "p = [";
		for (auto i: p)
		std::cout << i << ", ";
		cout << "]" << endl;

		cout << "pp = " << pp << endl;
		cout << "zeta = " << zeta << endl;

		}
*/

		for(int i=0; i<N; i++){

			//if(i%1==0) cout << i << endl;
			if(fixed_Z==true) break;
       
			if(abs(zeta-1)<1E-5) {
				if(it%sampling_rate==0  &&  it>= Niter/10*5) sampling.push_back(Z[i]); 
				continue;
			}

			stop=false;

			double prob[K];
			double gg[K];
			double norm=0;
			double gmax=0;

			for(int k1=0; k1<K;k1++)	{	

				double g=0;

				if(use_Potts==true) {
			      		double n_in=0;
					for(int j=0; j<q;j++) {
						int index=Iin.at(q*i+j);
						if(Z[index]==k1) n_in=n_in+1.;
	      				}
					double m_in=0;
					for(int j=0; j<Iout_count[i];j++) {
						int index=Iout.at(Iout_track[i]+j); //Iout.at(N*i+j);
						if(index > -1 && Z[index]==k1) m_in=m_in+1.;
	      				}

					//cout << "nin " << n_in << " min " << m_in << " nn " << Iout_count[i] << endl;

	                		//g=pow(zeta/(1-zeta),n_in+m_in)/Zpart(N,NN[k1],zeta,q);
					g=(n_in+m_in)*log(zeta/(1-zeta))-log(Zpart(N,NN[k1],zeta,q));					
					
					//g=g*pow(Zpart(N,NN[k1]-1,zeta,q)/Zpart(N,NN[k1],zeta,q),NN[k1]-1);
					g=g+log(Zpart(N,NN[k1]-1,zeta,q)/Zpart(N,NN[k1],zeta,q))*(NN[k1]-1);

					//cout << "g" << g <<  endl;

				}
	
				if(g > gmax) gmax=g;

				gg[k1]=g;
				//cout << g << endl;

				//prob[k1]=p[k1]*d[k1]*pow(MU[i],-(d[k1]+1))*g;
				//prob[k1]=log(p[k1]*d[k1])-(d[k1]+1)*log(MU[i]);

				//norm+=prob[k1];

          		}

			for(int k1=0; k1<K;k1++)    gg[k1]=exp(gg[k1]-gmax); 

			for(int k1=0; k1<K;k1++)  {
				  prob[k1]=p[k1]*d[k1]*pow(MU[i],-(d[k1]+1))*gg[k1];
				  norm+=prob[k1];

				/*
				if(it==0){
				cout << "**" << i << k1 << endl;
				cout << prob[k1] << endl;
				cout << p[k1] << endl;
				cout << d[k1] << endl;
				cout << MU[k1] << endl;
				cout << gg[k1] << endl;
				}
				*/

			}

			/*
			if(it==0){
			cout << "prob is ";
			for (auto i: prob)
			std::cout << i << ' ';
			cout << endl;
			}
			*/


			for(int k1=0; k1<K;k1++)  prob[k1]=prob[k1]/norm;	

			while(stop==false){

				int r1 = rand()%K; //int between 0 and <K
				MyFile << r1 << ",";
				double r2 = double(rand())/RAND_MAX;
				MyFile << r2 << ",";
				
				if(prob[r1] > r2) {
					stop=true;
					if(it%sampling_rate==0  &&  it>= Niter/10*5) sampling.push_back(r1);
					NN[Z[i]]-=1;
					a1[Z[i]]-=1;
					c1[Z[i]]-=1;
					V[Z[i]]-=log(MU[i]);
					b1[Z[i]]-=log(MU[i]);
					Z[i]=r1;
					NN[Z[i]]+=1;
					a1[Z[i]]+=1; 
					c1[Z[i]]+=1;
					V[Z[i]]+=log(MU[i]);
					b1[Z[i]]+=log(MU[i]);

					/*
					if(it==0){
					cout << "i " << i << " ";
					cout << "r1 " << r1 << " ";
					cout << "r2 " << r2 << " ";
					cout << "gg is ";
					for (auto i: gg)
					std::cout << i << ' ';
					cout << "norm " << norm << " ";
					cout << "gmax " << gmax << " ";
					cout << "prob[r1] " << prob[r1] << " ";
					cout << endl;
					}
					*/
					//cout << r1 << " ";
				}

			}

		}

		//cout << endl;


	/****** updating prior on zeta *******************************************************************************************************/

		//cout << "f1[0] " << f1[0] << endl;
		//cout << "f1[1] " << f1[1] << endl;


                N_in=0;
                for(int i=0; i< N; i++) {
                        int k=Z[i];
                        for(int j=0; j<q;j++) {
                                int index=Iin.at(q*i+j);
                                if(Z[index]==k) N_in=N_in+1.;
                    }
                }

		f1[0]=f[0]+N_in;
        f1[1]=f[1]+N*q-N_in;
		//cout << "f1[0] " << f1[0] << endl;
		//cout << "f1[1] " << f1[1] << endl;
		//cout << "***" << endl;

/* likelihood ****************************************************************************************************/
/*
				if(it==0){
				cout << "****";
				for (auto i: p) cout << i << " ";
				cout << endl;
				for (auto i: d) cout << i << " ";
				cout << endl;
				for (auto i: NN) cout << i << " ";
				cout << endl;
				cout << zeta << endl;
				cout << q << endl;
				cout << "****";
				}
*/

		double lik0=0;
                for(int i=0; i< N; i++) lik0=lik0+log(p[Z[i]])+log(d[Z[i]])-(d[Z[i]]+1)*log(MU.at(i));
                
                double lik1=lik0+log(zeta/(1-zeta))*N_in;

                for(int k1=0; k1<K; k1++) lik1=lik1-NN[k1]*log(Zpart(N, NN[k1], zeta, q));
		
		cout << "Zpart(N, NN[1], zeta, q)" << Zpart(N, NN[1], zeta, q) << endl;
		//cout << lik0 << " " << lik1 << endl;


	/* save data **********************************************************************************************************/
	

		if(it%sampling_rate==0  &&  it>= Niter/10*5) sampling.push_back(lik0);
		if(it%sampling_rate==0  &&  it>= Niter/10*5) sampling.push_back(lik1);

	}

    for (auto i: sampling){
    std::cout << i << ' ';
	MyFile << i << ",";
	}
    cout << endl;

	


    return 0;
}



