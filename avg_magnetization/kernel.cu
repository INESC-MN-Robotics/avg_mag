#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <cstring>  
#include <string>
#include <sstream>  
#include <thrust/reduce.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include "cu_veclib.cuh"

using namespace std;

__device__ double intg = 1.17318;
__device__ double Ms = 800;
__device__ double Vcell = 8e-18, Acell = 4e-12;
__device__ int d_ni = 42, d_nj = 17;


__global__ void HMj(double *HMj1, vec H, vec *Mj){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	HMj1[i] = H * Mj[i];
}

__global__ void kMj(double *kMj1, vec *Mj){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	vec K(1000, 0, 0);
	if (i%d_ni != (d_ni - 1) % d_ni && i%d_ni != 0 && i > d_ni && i < (d_nj - 1)*d_ni)
		kMj1[i] = K.abs()*(cross(K.norm(), Mj[i].norm())) ^ 2;
}

__global__ void exch(double *exch1, vec *Mi){
	double J = 100;
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i%d_ni != (d_ni - 1) % d_ni && i%d_ni != 0 && i > d_ni && i < (d_nj - 1)*d_ni)
		exch1[i] = J*(Mi[i + 1].norm() + Mi[i - 1].norm() + Mi[i + d_ni].norm() + Mi[i - d_ni].norm())*Mi[i].norm();
}

__global__ void demag(double *demag1, vec *Mj){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	vec nx1(1, 0, 0), nx_1(-1, 0, 0), ny1(0, 1, 0), ny_1(0, -1, 0);

	if (i%d_ni != (d_ni - 1) % d_ni && i%d_ni != 0 && i > d_ni && i < (d_nj - 1)*d_ni)
		demag1[i] = .5*Mj[i] * ((Mj[i + 1] * nx1)*nx1*intg + (Mj[i - 1] * nx_1)*nx_1*intg + (Mj[i + d_ni] * ny1)*ny1*intg + (Mj[i - d_ni] * ny_1)*ny_1*intg);
}

__global__ void energy(double *energy1, double *HMj, double *kMj, double *exch, double *demag){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	energy1[i] = -HMj[i] + kMj[i] - exch[i] - demag[i];
}

__global__ void init_doubles(double *Ej, double *HMj, double *kMj, double *exch, double *demag){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	Ej[i] = 0;
	HMj[i] = 0;
	kMj[i] = 0;
	exch[i] = 0;
	demag[i] = 0;
}

__global__ void init_Mi(vec *Mi, curandState *s, double *theta, double theta_d){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	Mi[i].x = 0;
	Mi[i].y = 0;
	Mi[i].z = 0;
	theta[i] = curand_uniform_double(&s[i])*3.14159*((double)2);
	if (i%d_ni != (d_ni - 1) % d_ni && i%d_ni != 0 && i > d_ni && i < (d_nj - 1)*d_ni){
		Mi[i].x = Ms*cos(-.3/*theta[i]*/);
		Mi[i].y = Ms*sin(-.3/*theta[i]*/);
	}
}

__global__ void define_Mi(vec *Mi, vec *Mi_avg){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i%d_ni != (d_ni - 1) % d_ni && i%d_ni != 0 && i > d_ni && i < (d_nj - 1)*d_ni)
		Mi[i] = Mi_avg[0];
}

__global__ void init_rand(curandState *s){
	unsigned int seed = (unsigned int)clock();
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	curand_init(seed, i, 0, &s[i]);
}

__global__ void d_rand(vec *Mi, curandState *s, double *theta, double *r){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i%d_ni != (d_ni - 1) % d_ni && i%d_ni != 0 && i > d_ni && i < (d_nj - 1)*d_ni){
		r[i] = curand_uniform_double(&s[i]);
		if (Mi[i].y > 0 && Mi[i].x > 0)
			theta[i] = atan(Mi[i].y / Mi[i].x);
		if (Mi[i].y < 0 && Mi[i].x > 0)
			theta[i] = atan(Mi[i].y / Mi[i].x);
		if (Mi[i].y > 0 && Mi[i].x < 0)
			theta[i] = atan(Mi[i].y / Mi[i].x) + 3.14159;
		if (Mi[i].y < 0 && Mi[i].x < 0)
			theta[i] = atan(Mi[i].y / Mi[i].x) + 3.14159;
		if (r[i]<.5)
			theta[i] = theta[i] + curand_uniform_double(&s[i])*3.14159 / ((double)60);
		else
			theta[i] = theta[i] - curand_uniform_double(&s[i])*3.14159 / ((double)60);
		Mi[i].x = Ms*cos(theta[i]);
		Mi[i].y = Ms*sin(theta[i]);
		Mi[i].z = 0;
	}
}

__global__ void stripes(vec *Mi){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j;
	if (i%d_ni != (d_ni - 1) % d_ni && i%d_ni != 0 && i > d_ni && i < (d_nj - 1)*d_ni){
		for (j = 1; j<d_nj; j++){
			if (i > j*d_ni && i < (j + 1)*d_ni - 1)
				Mi[i] = Mi[j*d_ni + 1];

		}
	}
}

bool thermal(double Et_p, double Et, double kB, double Vcell){
	double a, b;
	a = exp((Et_p - Et) * Vcell / kB);
	b = (Et_p - Et) * Vcell / kB;
	/* Check for overflow*/
	if (log(DBL_MAX) < (Et_p - Et) * Vcell / kB)
		return true;
	else if (log(DBL_MIN) >= (Et_p - Et) * Vcell / kB)
		return false;
	else if ((Et_p - Et) == 0)
		return false;

	/*do calculation and return "true" or "false"*/
	else{
		if (a < rand() / RAND_MAX)
			return false;
		else
			return true;
	}
}


int main(){

	int ni = 42; //# de threads (colunas da matriz)
	int nj = 17; //# de blocos (linhas da matriz)
	int i, j = 0, k = 0, l = 0;
	bool a = 1;

	int threadsPerBlock = ni;
	int BlocksPerGrid = nj;

	vec *h_Mi, *h_Mi_p;
	vec Mi_avg(0, 0, 0), h_H;
	double *h_Ej, *h_kMj, *h_Hmj, *h_exch, *h_demag;
	double Ms = 800;
	long double Et_p = 1e10, Et = 1e11, kB = 3.7674e-14, Vcell = 8e-18;


	//---------------------------ALOCAÇÃO DE ESPAÇO----------------------------//
	h_Mi = (vec*)malloc(sizeof(vec)*ni*nj);
	h_Mi_p = (vec*)malloc(sizeof(vec)*ni*nj);
	h_Ej = (double*)malloc(sizeof(double)*ni*nj);
	h_kMj = (double*)malloc(sizeof(double)*ni*nj);
	h_Hmj = (double*)malloc(sizeof(double)*ni*nj);
	h_exch = (double*)malloc(sizeof(double)*ni*nj);
	h_demag = (double*)malloc(sizeof(double)*ni*nj);

	vec *d_Mi, *d_Mi_avg, *d_H;
	vec Mi_avg_p;
	double *d_Ej, *d_kMj, *d_Hmj, *d_exch, *d_demag;
	double *theta, *r;
	curandState *s;
	string nficheiro;

	cudaMalloc(&d_H, sizeof(vec));
	cudaMalloc(&d_Mi, sizeof(vec)*ni*nj);
	cudaMalloc(&d_Ej, sizeof(double)*ni*nj);
	cudaMalloc(&d_kMj, sizeof(double)*ni*nj);
	cudaMalloc(&d_Hmj, sizeof(double)*ni*nj);
	cudaMalloc(&d_exch, sizeof(double)*ni*nj);
	cudaMalloc(&d_demag, sizeof(double)*ni*nj);
	cudaMalloc(&s, sizeof(curandState)*ni*nj);
	cudaMalloc(&theta, sizeof(double));
	cudaMalloc(&d_Mi_avg, sizeof(vec));
	cudaMalloc(&r, sizeof(double)*ni*nj);

	cout << "POWERED BY CUDA" << endl << endl;

	_sleep(2000);

	//-----------INICIALIZAÇÃO------------------//

	init_rand << <BlocksPerGrid, threadsPerBlock >> >(s);
	init_doubles << <BlocksPerGrid, threadsPerBlock >> >(d_Ej, d_Hmj, d_kMj, d_exch, d_demag);
	init_Mi << <BlocksPerGrid, threadsPerBlock >> >(d_Mi, s, theta, 1.57);
	cudaMemcpy(h_Mi, d_Mi, sizeof(vec)*ni*nj, cudaMemcpyDeviceToHost);

	for (i = 0; i < ni*nj; i++){
		Mi_avg = Mi_avg + h_Mi[i];
	}
	Mi_avg = Mi_avg*((double)1 / ((ni - 2)*(nj - 2)));

	Mi_avg_p = Mi_avg;
	Mi_avg = Ms*Mi_avg.norm();

	cudaMemcpy(d_Mi_avg, &Mi_avg, sizeof(vec), cudaMemcpyHostToDevice);

	define_Mi << <BlocksPerGrid, threadsPerBlock >> >(d_Mi, d_Mi_avg);
	d_rand << <BlocksPerGrid, threadsPerBlock >> >(d_Mi, s, theta, r);
	stripes << <BlocksPerGrid, threadsPerBlock >> >(d_Mi);

	//cudaDeviceSynchronize();

	////cudaMemcpy(d_H, &h_H, sizeof(vec), cudaMemcpyHostToDevice);
	//HMj << <BlocksPerGrid, threadsPerBlock >> >(d_Hmj, h_H, d_Mi);
	//kMj << <BlocksPerGrid, threadsPerBlock >> >(d_kMj, d_Mi);
	//exch << <BlocksPerGrid, threadsPerBlock >> >(d_exch, d_Mi);
	//demag << <BlocksPerGrid, threadsPerBlock >> >(d_demag, d_Mi);
	//energy << <BlocksPerGrid, threadsPerBlock >> >(d_Ej, d_Hmj, d_kMj, d_exch, d_demag);

	thrust::device_ptr<double> d_energy_thrust = thrust::device_pointer_cast(d_Ej);

	//Et = thrust::reduce(d_energy_thrust, d_energy_thrust + ni*nj);

	//Et_p = Et;

	cout << "Introduzir valor de Hx" << endl;
	cin >> h_H.x;
	cout << "Introduzir valor de Hy" << endl;
	cin >> h_H.y;

	cout << "Foi introduzido Hx=" << h_H.x << " e Hy=" << h_H.y << endl;

	cout << "Introduza o nome do ficheiro" << endl;
	cin >> nficheiro;

	_sleep(2000);

	ofstream tofile(nficheiro);
	//ofstream tofile2("Mi_x_2_d_term.txt");
	srand(time(NULL));

	k = 0;
	Et_p = 1e10;
	Et = 1e11;
	//cudaMemcpy(d_H, &h_H, sizeof(vec), cudaMemcpyHostToDevice);
	cout << h_H.x << endl;
	while (k != 10){
		while (false == thermal(Et_p, Et, kB, Vcell)/*Et_p<=Et*/){
			l++;
			cudaDeviceSynchronize();
			d_rand << <BlocksPerGrid, threadsPerBlock >> >(d_Mi, s, theta, r);
			stripes << <BlocksPerGrid, threadsPerBlock >> >(d_Mi);
			HMj << <BlocksPerGrid, threadsPerBlock >> >(d_Hmj, h_H, d_Mi);
			kMj << <BlocksPerGrid, threadsPerBlock >> >(d_kMj, d_Mi);
			exch << <BlocksPerGrid, threadsPerBlock >> >(d_exch, d_Mi);
			demag << <BlocksPerGrid, threadsPerBlock >> >(d_demag, d_Mi);
			energy << <BlocksPerGrid, threadsPerBlock >> >(d_Ej, d_Hmj, d_kMj, d_exch, d_demag);

			Et = thrust::reduce(d_energy_thrust, d_energy_thrust + ni*nj);

			if (l == 1000){
				//cout << "Nao convergiu :(" << endl;
				a = false;
				break;
			}
			a = 1;
		}
		if (a == 1){
			Et_p = Et;
			cudaMemcpy(h_Mi, d_Mi, sizeof(vec)*ni*nj, cudaMemcpyDeviceToHost);
			h_Mi_p = h_Mi;

			for (i = 0; i < ni*nj; i++){
				Mi_avg = Mi_avg + h_Mi[i];
			}

			Mi_avg = Mi_avg*((double)1 / ((ni - 2)*(nj - 2)));

			if (abs(Mi_avg.x - Mi_avg_p.x) < .05 && abs(Mi_avg.y - Mi_avg_p.y) < .05)
				break;

			Mi_avg_p = Mi_avg;
			Mi_avg = Ms*Mi_avg_p.norm();

			cudaMemcpy(d_Mi_avg, &Mi_avg, sizeof(vec), cudaMemcpyHostToDevice);
			define_Mi << <BlocksPerGrid, threadsPerBlock >> >(d_Mi, d_Mi_avg);
			//cudaMemcpy(h_Mi, d_Mi, sizeof(vec)*ni*nj, cudaMemcpyDeviceToHost);
			k = 0;
			cout << Mi_avg << endl;
		}
		else{
			k++;
			cout << k << endl;
			cudaMemcpy(d_Mi, h_Mi, sizeof(vec)*ni*nj, cudaMemcpyHostToDevice);
		}
		cudaDeviceSynchronize();
		l = 0;
		//cudaMemcpy(d_Mi, h_Mi, sizeof(vec)*ni*nj, cudaMemcpyHostToDevice);
		Mi_avg.x = 0;
		Mi_avg.y = 0;
		Mi_avg.z = 0;
	}
	//tofile << h_H.x << "\t" << Mi_avg_p << endl;
	//cudaMemcpy(h_Mi, d_Mi, sizeof(vec)*ni*nj, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_Hmj, d_Hmj, sizeof(double)*ni*nj, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_kMj, d_kMj, sizeof(double)*ni*nj, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_exch, d_exch, sizeof(double)*ni*nj, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_demag, d_demag, sizeof(double)*ni*nj, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_Ej, d_Ej, sizeof(double)*ni*nj, cudaMemcpyDeviceToHost);


	////init_rand << <BlocksPerGrid, threadsPerBlock >> >(s);
	////d_rand << <BlocksPerGrid, threadsPerBlock >> >(teste, s);

	////init_Mi << <BlocksPerGrid, threadsPerBlock >> >(theta, ni, nj);


	cudaMemcpy(h_Hmj, d_Hmj, sizeof(double)*ni*nj, cudaMemcpyDeviceToHost);
	for (i = 0; i < nj; i++){
		for (j = 0; j < ni; j++)
			tofile << i << "\t" << j << "\t" << h_Mi_p[j + i*ni] << endl;
	}

	tofile.close();

	//cudaMemcpy(teste1, teste, sizeof(double)*ni*nj, cudaMemcpyDeviceToHost);


	//----------------------------Libertação de espaço-----------------------------//
	cudaFree(d_Mi);
	cudaFree(d_Ej);
	cudaFree(d_kMj);
	cudaFree(d_Hmj);
	cudaFree(d_exch);
	cudaFree(d_demag);
	cudaFree(theta);
	cudaFree(s);

	free(h_Mi);
	free(h_Ej);
	free(h_kMj);
	free(h_Hmj);
	free(h_exch);
	free(h_demag);

	cout << "PROGRAMA CORRIDO COM SUCESSO!" << endl << "Prima qualquer tecla para sair..." << endl;
	cin.ignore();
	cin.get();
	return 0;
}
