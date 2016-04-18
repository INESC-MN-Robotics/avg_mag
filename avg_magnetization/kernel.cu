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

#define PI = 3.14159

using namespace std;


__global__ void init_map(vec *d_coord){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	if (i < 128 && j < 128){
		d_coord[i + j * 128].x = i*0.001 / (double)128;
		d_coord[i + j * 128].y = j*0.001 / (double)128;
		d_coord[i + j * 128].z = 0;
	}
}

__global__ void init_pos(vec *d_pil_pos, double x_off, double y_off){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < 50){
		d_pil_pos[i].x = d_pil_pos[i].x + x_off;
		d_pil_pos[i].y = d_pil_pos[i].y + x_off;
	}
}

__global__ void init_dist(vec *d_pil_pos, vec *d_coord, vec *d_dist){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int k;
	if (i < 128 && j < 128){
		for (k = 0; k < 50; k++)
			d_dist[k + 50 * i + 50 * 128 * j] = d_coord[i + 128 * j] - d_pil_pos[k];
	}
}
__global__ void calc_H(vec *d_dist, vec *d_dip, vec *d_Hi_inc, vec *d_Hi_tot, int *d_keys){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int k;
	if (i < 128 && j < 128){
		for (k = 0; k < 50; k++)
			d_Hi_inc[k + 50 * i + 50 * 128 * j] = (double)3 * d_dist[k + 50 * i + 50 * 128 * j] * (d_dip[k] * d_dist[k + 50 * i + 50 * 128 * j]) / pow(d_dist[k + 50 * i + 50 * 128 * j].abs(), 5) - d_dip[k] / pow(d_dist[k + 50 * i + 50 * 128 * j].abs(), 3);
	}
	if (i < 128 * 128 * 50){
		d_keys[i] = 50;
	}
}


int main(){
	// PLACA GRÁFICA: NVIDIA GEFORCE 820M - COMPUTE CAPABILITY: 2.1

	int ndip = 50; //# de dipolos
	int ni = 128; //# de threads (linhas da matriz)
	int nj = 128*ndip; //# de blocos (linhas da matriz*dipolos do pilar = 6400)
	int i, j = 0, k = 0;
	string fileplace_def, fileplace_vec, fileplace_points, fileplace_avgs;

	int threadsPerBlock = ni;
	int BlocksPerGrid = nj;
	double x_off=0.0005, y_off=0.0005;

	vec *h_dip, *h_pil_pos, *h_Hi_inc, *temp_dip;
	vec h_Hi_avg(0, 0, 0);


	//---------------------------ALOCAÇÃO DE ESPAÇO----------------------------//
	h_dip = (vec*)malloc(sizeof(vec)*ndip); //Vector de magnetização dos dipolos
	temp_dip = (vec*)malloc(sizeof(vec)*10); //Vector auxiliar
	h_pil_pos = (vec*)malloc(sizeof(vec)*ndip); //Posição dos dipolos
	h_Hi_inc = (vec*)malloc(sizeof(vec)*ni*ni); //Valor do campo incidente num elemento de área 
	
	int *d_keys, *d_rest;
	vec *d_dist, *d_pil_pos, *d_Hi_inc, *d_dip, *d_coord, *d_Hi_tot;
	vec *d_Hi_avg;

	cudaMalloc(&d_Hi_avg, sizeof(vec));
	cudaMalloc(&d_dist, sizeof(vec)*ni*nj);
	cudaMalloc(&d_keys, sizeof(int)*ni*nj);
	cudaMalloc(&d_rest, sizeof(int));
	cudaMalloc(&d_pil_pos, sizeof(vec)*ndip);
	cudaMalloc(&d_Hi_inc, sizeof(vec)*ni*nj);
	cudaMalloc(&d_Hi_tot, sizeof(vec)*ni*ni);
	cudaMalloc(&d_dip, sizeof(vec)*ndip);
	cudaMalloc(&d_coord, sizeof(vec)*ni*ni);

	thrust::device_ptr<vec> Hi_tot_thrust = thrust::device_pointer_cast(d_Hi_tot);
	thrust::device_ptr<vec> Hi_inc_thrust = thrust::device_pointer_cast(d_Hi_inc);
	thrust::device_ptr<int> keys_thrust = thrust::device_pointer_cast(d_keys);
	thrust::device_ptr<int> rest_thrust = thrust::device_pointer_cast(d_rest);

	cout << "POWERED BY CUDA" << endl << endl;

	_sleep(2000);

	fstream fileout_avgs;
	fileplace_avgs = "C:\\Users\\Pedro\\Documents\\MATLAB\\dados\\avgs.txt";
	fileout_avgs.open(fileplace_avgs);

	for (i = 1; i < 2; i++){

		h_Hi_avg.x = 0;
		h_Hi_avg.y = 0;
		h_Hi_avg.z = 0;

		//------- PARSING (Entrada de dados) -------//

		// O delimitador é qq caracter que não seja um número válido (para ints e floating points de qq precisão)

		fileplace_def = "C:\\Users\\Pedro\\Documents\\MATLAB\\dados\\defs_"+to_string(i)+".txt";
		fileplace_vec = "C:\\Users\\Pedro\\Documents\\MATLAB\\dados\\vecs_"+to_string(i)+".txt";

		fstream filein_def;
		fstream filein_vec;
		filein_def.open(fileplace_def);
		filein_vec.open(fileplace_vec);

		for (j = 0; j < 10; j++){
			filein_vec >> temp_dip[j].x >> temp_dip[j].y >> temp_dip[j].z;
		}

		filein_vec.close();

		for (j = 0; j < ndip; j++){
			k++;
			if (k == 10)
				k = 0;
			filein_def >> h_pil_pos[j].x >> h_pil_pos[j].y >> h_pil_pos[j].z;
			h_dip[j] = temp_dip[k];
		}

		filein_def.close();

		//------ DATA TRANSFER (H -> D) --------//

		cudaMemcpy(d_dip, h_dip, sizeof(vec)*ndip, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pil_pos, h_pil_pos, sizeof(vec)*ndip, cudaMemcpyHostToDevice);

		//-----------INICIALIZAÇÃO------------------//

		init_map << <BlocksPerGrid, threadsPerBlock >> >(d_coord);
		init_pos << <BlocksPerGrid, threadsPerBlock >> >(d_pil_pos, x_off, y_off);
		init_dist << <BlocksPerGrid, threadsPerBlock >> >(d_pil_pos, d_coord, d_dist);

		//------ CÁLCULO ------ //
		calc_H << <BlocksPerGrid, threadsPerBlock >> >(d_dist, d_dip, d_Hi_inc, d_Hi_tot, d_keys);
		thrust::reduce_by_key(keys_thrust, keys_thrust + ni * nj, Hi_inc_thrust, rest_thrust, Hi_tot_thrust);

		//------ DATA TRANSFER (D -> H) --------//

		//cudaMemcpy(h_dist, d_Hi_inc, sizeof(vec)*ni*nj, cudaMemcpyDeviceToHost); //UNCOMMENT TO DIAGNOSE
		fstream fileout_points;
		fileplace_points = "C:\\Users\\Pedro\\Documents\\CUDA_OUT\\points_" + to_string(i) + ".txt";
		fileout_points.open(fileplace_points);
		for (i = 0; i < ni; i++){
			for (j = 0; j < ni; j++){
				fileout_points << Hi_tot_thrust[i];
				h_Hi_avg = h_Hi_avg + Hi_tot_thrust[i];
			}
			fileout_points << endl;
		}
		h_Hi_avg = h_Hi_avg / (128 * 128);
		fileout_points.close();
		fileout_avgs << h_Hi_avg << endl;
	}

	fileout_avgs.close();

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
	free(temp_dip);
	
	cout << "PROGRAMA CORRIDO COM SUCESSO!" << endl << "Prima qualquer tecla para sair..." << endl;
	cin.get();
	return 0;
}
