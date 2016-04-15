#include "cu_veclib.cuh"
#include <cmath>
#include <iostream>
#include <thrust/reduce.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>
#include <cuda.h>

__host__ __device__ vec::vec(){
	x = 0;
	y = 0;
	z = 0;
}
__host__ __device__ vec::vec(double x1, double x2, double x3){
	x = x1;
	y = x2;
	z = x3;
}
__host__ __device__  double vec::abs(){
	return sqrt(x*x + y*y + z*z);
}

__host__ __device__ vec vec::norm(){
	vec temp;
	if (x == 0 && y == 0 && z == 0)
		temp.x = 1e-50;
	else{
		temp.x = x / sqrt(x*x + y*y + z*z);
		temp.y = y / sqrt(x*x + y*y + z*z);
		temp.z = z / sqrt(x*x + y*y + z*z);
	}
	return temp;
}

__host__ __device__ vec& vec::operator=(const vec& seg){
	x = seg.x;
	y = seg.y;
	z = seg.z;
	return *this;
}

std::ostream& operator<<(std::ostream& os, const vec& out){
	os << out.x << "\t" << out.y << "\t" << out.z << endl;
	return os;
}

__host__ __device__ vec operator+(const vec &pri, const vec &seg){
	vec temp;
	temp.x = pri.x + seg.x;
	temp.y = pri.y + seg.y;
	temp.z = pri.z + seg.z;
	return(temp);
}
__host__ __device__ vec operator*(const vec &a, double b)
{
	vec d;

	d.x = a.x*b;
	d.y = a.y*b;
	d.z = a.z*b;

	return d;
}
__host__ __device__ vec operator*(double a, const vec &b)
{
	vec d;

	d.x = b.x*a;
	d.y = b.y*a;
	d.z = b.z*a;

	return d;
}
__host__ __device__ double operator*(const vec &a, const vec &b)
{
	double d;

	d = a.x*b.x + a.y*b.y + a.z*b.z;

	return d;
}
__host__ __device__ vec operator-(const vec &a, const vec &b){
	vec d;
	d.x = a.x - b.x;
	d.y = a.y - b.y;
	d.z = a.z - b.z;
	return d;
}
__host__ __device__ vec cross(const vec &a, const vec &b){
	vec d;
	d.x = a.y*b.z - a.z*b.y;
	d.y = a.z*b.x - a.x*b.z;
	d.z = a.x*b.y - a.y*b.x;
	return d;
}
__host__ __device__ double operator^(const vec &a, double b){
	double d;
	d = pow(sqrt(a.x*a.x + a.y*a.y + a.z*a.z), b);
	return(d);
}