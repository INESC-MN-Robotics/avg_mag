#include <iostream>
#include <fstream>
#include <string>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include "cu_veclib.cuh"

#define PI 3.14159
#define DIPOLES 90
#define MOMENTO .12
#define DIMENSAOX 0.003
#define DIMENSAOY 0.003
#define REP 25

using namespace std;


__global__ void init_map(vec *d_coord){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	if (i < 128 && j < 128){
		d_coord[i + j * 128].x = i*DIMENSAOX / (double)128;
		d_coord[i + j * 128].y = j*DIMENSAOY / (double)128;
		d_coord[i + j * 128].z = 0;
	}
}

__global__ void init_pos(vec *d_pil_pos, double x_off, double y_off){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < DIPOLES){
		d_pil_pos[i].x = d_pil_pos[i].x + x_off;
		d_pil_pos[i].y = d_pil_pos[i].y + y_off;
	}
}

__global__ void init_dist(vec *d_pil_pos, vec *d_coord, vec *d_dist){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int k;
	for (k = 0; k < DIPOLES; k++){
		if (i < 128 && j < 128)
			d_dist[k + DIPOLES * i + DIPOLES * 128 * j] = d_coord[i + 128 * j] - d_pil_pos[k];
	}
}
__global__ void calc_H(vec *d_dist, vec *d_dip, vec *d_Hi_inc, vec *d_Hi_tot, int *d_keys){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int k;
	for (k = 0; k < DIPOLES; k++){
		if (i < 128 && j < 128){
			d_Hi_inc[k + DIPOLES * i + DIPOLES * 128 * j] = (double)3 * d_dist[k + DIPOLES * i + DIPOLES * 128 * j] * (d_dip[k] * d_dist[k + DIPOLES * i + DIPOLES * 128 * j]) / pow(d_dist[k + DIPOLES * i + DIPOLES * 128 * j].abs(), 5) - d_dip[k] / pow(d_dist[k + DIPOLES * i + DIPOLES * 128 * j].abs(), 3);
			d_Hi_inc[k + DIPOLES * i + DIPOLES * 128 * j] = d_Hi_inc[k + DIPOLES * i + DIPOLES * 128 * j] * 0.001;
		}
	}
}

__global__ void index_keys(int *d_keys){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < 128 * 128 * DIPOLES)
		d_keys[i] = 1 + i / (int)DIPOLES;
}


int main(){
	// PLACA GRÁFICA: NVIDIA QUADRO P2000 - COMPUTE CAPABILITY: 6.1

	clock_t at, bt;
	int ndip = DIPOLES; //# de dipolos
	//ELEMENTOS DE SUPERFICIE
	int ni = 128; //X
	int nj = 128; //Y
	int nk = 15622; //DIPOLOS
	
	int i, j = 0, k = 0, f;
	string fileplace_def, fileplace_vec, fileplace_points, fileplace_avgs;

	int threadsPerBlock = ni;
	int BlocksPerGrid = nj;
	dim3 tpb(32, 32);
	dim3 bpg(4, 4, nk);
	double x_off=0.0015, y_off=0.0015;

	vec *h_dip, *h_pil_pos;
	vec **h_Hi_inc;
	vec h_Hi_avg(0, 0, 0);
	vec temp_dip(0, 0, 0); //temporary vector

	//---------------------------ALOCAÇÃO DE ESPAÇO----------------------------//
	h_dip = (vec*)malloc(sizeof(vec)*nk); //Vector de magnetização dos dipolos
	h_pil_pos = (vec*)malloc(sizeof(vec)*nk); //Posição dos dipolos
	h_Hi_inc = (vec**)malloc(sizeof(vec*)*ni);	//Valor do campo incidente num elemento de área 
	for(i = 0; i < ni; i++){
		h_Hi_inc[i] = (vec*)malloc(sizeof(vec)*nj);
	}
	
	int *d_keys, *d_rest;
	vec *d_dist, *d_pil_pos, *d_dip, *d_coord, *d_Hi_tot;
	vec *d_Hi_avg;
	vec **d_Hi_inc;

	cudaMalloc(&d_Hi_avg, sizeof(vec));
	cudaMalloc(&d_dist, sizeof(vec)*ni*nj);
	cudaMalloc(&d_keys, sizeof(int)*ni*nj);
	cudaMalloc(&d_rest, sizeof(int)*ni*ni);
	cudaMalloc(&d_pil_pos, sizeof(vec)*ndip);
	cudaMalloc(&d_Hi_tot, sizeof(vec)*ni*ni);
	cudaMalloc(&d_dip, sizeof(vec)*ndip);
	cudaMalloc(&d_coord, sizeof(vec)*ni*ni);
	cudaMalloc((vec**)&d_Hi_inc, sizeof(vec*)*ni);
	for(i = 0; i < ni; i++){
		cudaMalloc(&(d_Hi_inc[i]), sizeof(vec)*nj);
	}

	thrust::device_ptr<vec> Hi_tot_thrust = thrust::device_pointer_cast(d_Hi_tot);
	thrust::device_ptr<vec> Hi_inc_thrust = thrust::device_pointer_cast(d_Hi_inc);
	thrust::device_ptr<int> keys_thrust = thrust::device_pointer_cast(d_keys);
	thrust::device_ptr<int> rest_thrust = thrust::device_pointer_cast(d_rest);

	cout << "Simulador de nanorods" << endl << endl;
	
	at = clock();


	
	//--------------------------PARSING (ENTRADA DE DADOS)--------------------------//

	fileplace_def = "teste_displacement";
	fstream filein_def;
	filein_def.open(fileplace_def.c_str());
	
	for(i = 0; i < nk; i++){
		filein_def >> h_pil_pos[i].x >> h_pil_pos[i].y >> h_pil_pos[i].z >> temp_dip.x >> temp_dip.y >> temp_dip.z;
		h_pil_pos[i] = h_pil_pos[i] + temp_dip;
	}
	
	for(i = 0; i < nk-REP; i++){
		h_dip[i] = h_pil_pos[i+REP] - h_pil_pos[i];
		h_dip[i].norm();
	}
	for(i = nk - REP; i < nk; i++){
		h_dip[i] = h_dip[i-REP];
	}
	
	//----------------------------Libertação de espaço-----------------------------//
	
	cudaFree(d_Hi_avg);
	cudaFree(d_dist);
	cudaFree(d_pil_pos);
	cudaFree(d_Hi_inc);
	cudaFree(d_dip);
	cudaFree(d_coord);
	cudaFree(d_keys);
	cudaFree(d_rest);
	cudaFree(d_Hi_tot);

	free(h_dip);
	free(h_pil_pos);
	free(h_Hi_inc);

	at = (clock() - at) / CLOCKS_PER_SEC;
	
	cout << "PROGRAMA CORRIDO COM SUCESSO EM " << (double)at + (double)bt << " SEGUNDOS!" << endl;
	cout << "Tempo de processamento de calculo: " << (double)bt << " segundos." << endl << "Tempo de processamento de I/O: " << at << " segundos." << endl << "Prima qualquer tecla para sair..." << endl;
	cin.get();
	return 0;
}
