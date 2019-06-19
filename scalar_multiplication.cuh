#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "struct_types.cuh"

namespace ScalarMultiply {
	/**
	Perform scalar multiplication of points on the same curve in parallel. 
	For each element i: resultArray[i] = scalarArray[i] * pointArray[i].
	@param resultArray Pointer to array of affine points of given size where the result will be copied to.
	@param scalarArray Pointer to array of BigNum scalar values by which multiplication will be performed.
	@param pointArray Pointer to array of affine points of given size which are being multiplied. 
	@param count total number of elements.
	@param curveType curve on which points are located.
	@returns cudaError_t produced after launching the kernel.
	*/
	template <size_t N>
	__host__ cudaError_t scalarMultiplyOnGpu(AffinePoint<N> *resultArray, BigNum<N> *scalarArray,
		AffinePoint<N> *pointArray, unsigned int count, CurveType curveType);
}