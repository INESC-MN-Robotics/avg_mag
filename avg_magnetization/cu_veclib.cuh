#include <iostream>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>
#include <cuda.h>
#ifndef VECLIB
#define VECLIB

using namespace std;

class vec{
public:
	double x, y, z;
	__host__ __device__ vec();
	__host__ __device__ vec(double x1, double x2, double x3);
	__host__ __device__  double abs();
	__host__ __device__ vec norm();
	friend vec operator+(const vec&, const vec&);
	friend vec operator*(double, const vec&);
	friend vec operator*(const vec&, double);
	friend double operator*(const vec&, const vec&);
	friend vec operator-(const vec&, const vec&);
	friend vec cross(const vec&, const vec&);
	friend double operator^(const vec&, double);
	__host__ __device__ vec& operator=(const vec& seg);
	friend std::ostream& operator<<(std::ostream& os, const vec& out);
	friend __host__ __device__ vec operator+(const vec &pri, const vec &seg);
	friend __host__ __device__ vec operator*(const vec &a, double b);
	friend __host__ __device__ vec operator/(const vec &a, double b);
	friend __host__ __device__ vec operator*(double a, const vec &b);
	friend __host__ __device__ double operator*(const vec &a, const vec &b);
	friend __host__ __device__ vec operator-(const vec &a, const vec &b);
	friend __host__ __device__ vec cross(const vec &a, const vec &b);
	friend __host__ __device__ double operator^(const vec &a, double b);
};
#endif
