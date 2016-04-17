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
	if (i < 128*128 && j < 50){
		d_dist[i + 128 * 128 * j] = d_coord[i + 128 * 128 * j]-d_pil_pos[j];
	}
}
__global__ void calc_H(vec *d_dist, vec *d_dip, vec *d_Hi_inc, vec *d_Hi_tot, vec *d_Hi_temp, int *d_keys){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	if (i < 128 * 128 && j < 50){
		d_Hi_inc[i + 128 * 128 * j] = (double)3 * d_dist[i + 128 * 128 * j] * (d_dip[i + 128 * 128 * j] * d_dist[i + 128 * 128 * j]) / pow(d_dist[i + 128 * 128 * j].abs(), 5) - d_dip[i + 128 * 128 * j] / pow(d_dist[i + 128 * 128 * j].abs(),3);
		d_Hi_temp[j + 128 * 128 * i] = d_Hi_inc[i + 128 * 128 * j];
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
	int i, j = 0, k = 0, l = 0;
	bool a = 1;
	string fileplace_def, fileplace_vec;

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
	vec *d_dist, *d_pil_pos, *d_Hi_inc, *d_dip, *d_coord, *d_Hi_temp, *d_Hi_tot;
	vec *d_Mi_avg_p;

	cudaMalloc(&d_Mi_avg_p, sizeof(vec));
	cudaMalloc(&d_dist, sizeof(vec)*ni*nj);
	cudaMalloc(&d_keys, sizeof(int)*ni*nj);
	cudaMalloc(&d_rest, sizeof(int));
	cudaMalloc(&d_pil_pos, sizeof(vec)*ndip);
	cudaMalloc(&d_Hi_inc, sizeof(vec)*ni*nj);
	cudaMalloc(&d_Hi_tot, sizeof(vec)*ni*ni);
	cudaMalloc(&d_Hi_temp, sizeof(vec)*ni*nj);
	cudaMalloc(&d_dip, sizeof(vec)*ndip);
	cudaMalloc(&d_coord, sizeof(vec)*ni*ni);

	cout << "POWERED BY CUDA" << endl << endl;

	_sleep(2000);

	for (i = 1; i < 2; i++){

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
		calc_H << <BlocksPerGrid, threadsPerBlock >> >(d_dist, d_dip, d_Hi_inc, d_Hi_tot, d_Hi_temp, d_keys);
		thrust::device_ptr<vec> d_Hi_temp_thrust = thrust::device_pointer_cast(d_Hi_temp);
		thrust::device_ptr<vec> d_Hi_tot_thrust = thrust::device_pointer_cast(d_Hi_tot);
		thrust::device_ptr<int> d_keys_thrust = thrust::device_pointer_cast(d_keys);
		thrust::device_ptr<int> d_rest_thrust = thrust::device_pointer_cast(d_rest);
		thrust::reduce_by_key(d_keys_thrust, d_keys_thrust + 128 * 128, d_Hi_temp_thrust, d_rest_thrust, d_Hi_tot_thrust);
		
		//for (i = 0; i < ni*nj; i++){
		//	Mi_avg = Mi_avg + h_Mi[i];
		//}
		//Mi_avg = Mi_avg*((double)1 / ((ni - 2)*(nj - 2)));

		//Mi_avg_p = Mi_avg;
		//Mi_avg = Ms*Mi_avg.norm();

		//cudaMemcpy(d_Mi_avg, &Mi_avg, sizeof(vec), cudaMemcpyHostToDevice);

		//define_Mi << <BlocksPerGrid, threadsPerBlock >> >(d_Mi, d_Mi_avg);
		//d_rand << <BlocksPerGrid, threadsPerBlock >> >(d_Mi, s, theta, r);
		//stripes << <BlocksPerGrid, threadsPerBlock >> >(d_Mi);




		//------ DATA TRANSFER (D -> H) --------//


		//cudaDeviceSynchronize();

		////cudaMemcpy(d_H, &h_H, sizeof(vec), cudaMemcpyHostToDevice);
		//HMj << <BlocksPerGrid, threadsPerBlock >> >(d_Hmj, h_H, d_Mi);
		//kMj << <BlocksPerGrid, threadsPerBlock >> >(d_kMj, d_Mi);
		//exch << <BlocksPerGrid, threadsPerBlock >> >(d_exch, d_Mi);
		//demag << <BlocksPerGrid, threadsPerBlock >> >(d_demag, d_Mi);
		//energy << <BlocksPerGrid, threadsPerBlock >> >(d_Ej, d_Hmj, d_kMj, d_exch, d_demag);

		//thrust::device_ptr<double> d_energy_thrust = thrust::device_pointer_cast(d_Ej);

		//Et = thrust::reduce(d_energy_thrust, d_energy_thrust + ni*nj);

		//Et_p = Et;

		//cout << "Introduzir valor de Hx" << endl;
		//cin >> h_H.x;
		//cout << "Introduzir valor de Hy" << endl;
		//cin >> h_H.y;

		//cout << "Foi introduzido Hx=" << h_H.x << " e Hy=" << h_H.y << endl;

		//cout << "Introduza o nome do ficheiro" << endl;
		//cin >> nficheiro;

		//_sleep(2000);

		//ofstream tofile(nficheiro);
		////ofstream tofile2("Mi_x_2_d_term.txt");
		//srand(time(NULL));

		//k = 0;
		//Et_p = 1e10;
		//Et = 1e11;
		////cudaMemcpy(d_H, &h_H, sizeof(vec), cudaMemcpyHostToDevice);
		//cout << h_H.x << endl;
		//while (k != 10){
		//	while (false == thermal(Et_p, Et, kB, Vcell)/*Et_p<=Et*/){
		//		l++;
		//		cudaDeviceSynchronize();
		//		d_rand << <BlocksPerGrid, threadsPerBlock >> >(d_Mi, s, theta, r);
		//		stripes << <BlocksPerGrid, threadsPerBlock >> >(d_Mi);
		//		HMj << <BlocksPerGrid, threadsPerBlock >> >(d_Hmj, h_H, d_Mi);
		//		kMj << <BlocksPerGrid, threadsPerBlock >> >(d_kMj, d_Mi);
		//		exch << <BlocksPerGrid, threadsPerBlock >> >(d_exch, d_Mi);
		//		demag << <BlocksPerGrid, threadsPerBlock >> >(d_demag, d_Mi);
		//		energy << <BlocksPerGrid, threadsPerBlock >> >(d_Ej, d_Hmj, d_kMj, d_exch, d_demag);

		//		Et = thrust::reduce(d_energy_thrust, d_energy_thrust + ni*nj);

		//		if (l == 1000){
		//			//cout << "Nao convergiu :(" << endl;
		//			a = false;
		//			break;
		//		}
		//		a = 1;
		//	}
		//	if (a == 1){
		//		Et_p = Et;
		//		cudaMemcpy(h_Mi, d_Mi, sizeof(vec)*ni*nj, cudaMemcpyDeviceToHost);
		//		h_Mi_p = h_Mi;

		//		for (i = 0; i < ni*nj; i++){
		//			Mi_avg = Mi_avg + h_Mi[i];
		//		}

		//		Mi_avg = Mi_avg*((double)1 / ((ni - 2)*(nj - 2)));

		//		if (abs(Mi_avg.x - Mi_avg_p.x) < .05 && abs(Mi_avg.y - Mi_avg_p.y) < .05)
		//			break;

		//		Mi_avg_p = Mi_avg;
		//		Mi_avg = Ms*Mi_avg_p.norm();

		//		cudaMemcpy(d_Mi_avg, &Mi_avg, sizeof(vec), cudaMemcpyHostToDevice);
		//		define_Mi << <BlocksPerGrid, threadsPerBlock >> >(d_Mi, d_Mi_avg);
		//		//cudaMemcpy(h_Mi, d_Mi, sizeof(vec)*ni*nj, cudaMemcpyDeviceToHost);
		//		k = 0;
		//		cout << Mi_avg << endl;
		//	}
		//	else{
		//		k++;
		//		cout << k << endl;
		//		cudaMemcpy(d_Mi, h_Mi, sizeof(vec)*ni*nj, cudaMemcpyHostToDevice);
		//	}
		//	cudaDeviceSynchronize();
		//	l = 0;
		//	//cudaMemcpy(d_Mi, h_Mi, sizeof(vec)*ni*nj, cudaMemcpyHostToDevice);
		//	Mi_avg.x = 0;
		//	Mi_avg.y = 0;
		//	Mi_avg.z = 0;
		//}
		//tofile << h_H.x << "\t" << Mi_avg_p << endl;
		//cudaMemcpy(h_Mi, d_Mi, sizeof(vec)*ni*nj, cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_Hmj, d_Hmj, sizeof(double)*ni*nj, cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_kMj, d_kMj, sizeof(double)*ni*nj, cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_exch, d_exch, sizeof(double)*ni*nj, cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_demag, d_demag, sizeof(double)*ni*nj, cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_Ej, d_Ej, sizeof(double)*ni*nj, cudaMemcpyDeviceToHost);


		////init_rand << <BlocksPerGrid, threadsPerBlock >> >(s);
		////d_rand << <BlocksPerGrid, threadsPerBlock >> >(teste, s);

		////init_Mi << <BlocksPerGrid, threadsPerBlock >> >(theta, ni, nj);


		//cudaMemcpy(h_Hmj, d_Hmj, sizeof(double)*ni*nj, cudaMemcpyDeviceToHost);
		//for (i = 0; i < nj; i++){
		//	for (j = 0; j < ni; j++)
		//		tofile << i << "\t" << j << "\t" << h_Mi_p[j + i*ni] << endl;
		//}

		//tofile.close();

		//cudaMemcpy(teste1, teste, sizeof(double)*ni*nj, cudaMemcpyDeviceToHost);

	}


	//----------------------------Libertação de espaço-----------------------------//
	
	cudaFree(d_Mi_avg_p);
	cudaFree(d_dist);
	cudaFree(d_pil_pos);
	cudaFree(d_Hi_inc);
	cudaFree(d_dip);
	cudaFree(d_coord);

	free(h_dip);
	free(h_pil_pos);
	free(h_Hi_inc);
	
	cout << "PROGRAMA CORRIDO COM SUCESSO!" << endl << "Prima qualquer tecla para sair..." << endl;
	cin.get();
	return 0;
}
