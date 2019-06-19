#include "prime_curve_arithmetic.cuh"
#include "curve_definitions.cuh"
#include "scalar_multiplication.cuh"

#define min(a,b) a<b?a:b

namespace ScalarMultiply {

	namespace pca = PrimeCurveArithmetic;

	/*
	Helper function used inside scalarMultiplyOnGpu.
	*/
	template <size_t N>
	__host__ bool launchScalarMultKernel(AffinePoint<N> *dev_res, BigNum<N> *dev_scalar,
		AffinePoint<N> *dev_point, CurveType curve, unsigned int count);


	//Using manual definitions so that curve can be inlined during compile time.
	__global__ void scalarMultKernelSecp192r1(AffinePoint<6> *res, BigNum<6> *scalar,
		AffinePoint<6> *point, unsigned int count)
	{
		for (unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < count; idx += gridDim.x*blockDim.x) {
			res[idx] = pca::scalarMultBinary(scalar[idx], point[idx], &Curve::secp192r1);
		}
	}
	__global__ void scalarMultKernelSecp224r1(AffinePoint<7> *res, BigNum<7> *scalar,
		AffinePoint<7> *point, unsigned int count)
	{
		for (unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < count; idx += gridDim.x*blockDim.x) {
			res[idx] = pca::scalarMultBinary(scalar[idx], point[idx], &Curve::secp224r1);
		}
	}
	__global__ void scalarMultKernelSecp256r1(AffinePoint<8> *res, BigNum<8> *scalar,
		AffinePoint<8> *point, unsigned int count)
	{
		for (unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < count; idx += gridDim.x*blockDim.x) {
			res[idx] = pca::scalarMultBinary(scalar[idx], point[idx], &Curve::secp256r1);
		}
	}
	__global__ void scalarMultKernelSecp384r1(AffinePoint<12> *res, BigNum<12> *scalar,
		AffinePoint<12> *point, unsigned int count)
	{
		for (unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < count; idx += gridDim.x*blockDim.x) {
			res[idx] = pca::scalarMultBinary(scalar[idx], point[idx], &Curve::secp384r1);
		}
	}
	__global__ void scalarMultKernelSecp521r1(AffinePoint<17> *res, BigNum<17> *scalar,
		AffinePoint<17> *point, unsigned int count)
	{
		for (unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < count; idx += gridDim.x*blockDim.x) {
			res[idx] = pca::scalarMultBinary(scalar[idx], point[idx], &Curve::secp521r1);
		}
	}


	//Manual definitions to bypass compile time errors of incompatible arguments.
	__host__ bool launchScalarMultKernel(AffinePoint<6> *dev_res, BigNum<6> *dev_scalar,
		AffinePoint<6> *dev_point, CurveType curve, unsigned int count)
	{
		const int threadCount = 256;
		const int blockCount = min(count/threadCount+1,256);

		switch (curve)
		{
		case secp192r1:
			cudaFuncSetCacheConfig(scalarMultKernelSecp192r1, cudaFuncCachePreferL1);
			scalarMultKernelSecp192r1 << <blockCount, threadCount >> > (dev_res, dev_scalar, dev_point, count);
			break;
		default:
			fprintf(stderr, "No curve type found!/n");
			return 1;
		}

		return 0;
	}
	__host__ bool launchScalarMultKernel(AffinePoint<7> *dev_res, BigNum<7> *dev_scalar,
		AffinePoint<7> *dev_point, CurveType curve, unsigned int count)
	{
		const int threadCount = 256;
		const int blockCount = min(count / threadCount + 1, 256);

		switch (curve)
		{
		case secp224r1:
			cudaFuncSetCacheConfig(scalarMultKernelSecp224r1, cudaFuncCachePreferL1);
			scalarMultKernelSecp224r1 << <blockCount, threadCount >> > (dev_res, dev_scalar, dev_point, count);
			break;
		default:
			fprintf(stderr, "No curve type found!/n");
			return 1;
		}

		return 0;
	}
	__host__ bool launchScalarMultKernel(AffinePoint<8> *dev_res, BigNum<8> *dev_scalar,
		AffinePoint<8> *dev_point, CurveType curve, unsigned int count)
	{
		const int threadCount = 256;
		const int blockCount = min(count / threadCount + 1, 256);

		switch (curve)
		{
		case secp256r1:
			cudaFuncSetCacheConfig(scalarMultKernelSecp256r1, cudaFuncCachePreferL1);
			scalarMultKernelSecp256r1 << <blockCount, threadCount >> > (dev_res, dev_scalar, dev_point, count);
			break;
		default:
			fprintf(stderr, "No curve type found!/n");
			return 1;
		}

		return 0;
	}
	__host__ bool launchScalarMultKernel(AffinePoint<12> *dev_res, BigNum<12> *dev_scalar,
		AffinePoint<12> *dev_point, CurveType curve, unsigned int count)
	{
		const int threadCount = 256;
		const int blockCount = min(count / threadCount + 1, 256);

		switch (curve)
		{
		case secp384r1:
			cudaFuncSetCacheConfig(scalarMultKernelSecp384r1, cudaFuncCachePreferL1);
			scalarMultKernelSecp384r1 << <blockCount, threadCount >> > (dev_res, dev_scalar, dev_point, count);
			break;
		default:
			fprintf(stderr, "No curve type found!/n");
			return 1;
		}

		return 0;
	}
	__host__ bool launchScalarMultKernel(AffinePoint<17> *dev_res, BigNum<17> *dev_scalar,
		AffinePoint<17> *dev_point, CurveType curve, unsigned int count)
	{
		const int threadCount = 256;
		const int blockCount = min(count / threadCount + 1, 256);

		switch (curve)
		{
		case secp521r1:
			cudaFuncSetCacheConfig(scalarMultKernelSecp521r1, cudaFuncCachePreferL1);
			scalarMultKernelSecp521r1 << <blockCount, threadCount >> > (dev_res, dev_scalar, dev_point, count);
			break;
		default:
			fprintf(stderr, "No curve type found!/n");
			return 1;
		}

		return 0;
	}


	template <size_t N>
	__host__ cudaError_t scalarMultiplyOnGpu(AffinePoint<N> *resultArray, BigNum<N> *scalarArray,
		AffinePoint<N> *pointArray, unsigned int count, CurveType curveType)
	{
		AffinePoint<N> *dev_res = 0;
		BigNum<N> *dev_scalar = 0;
		AffinePoint<N> *dev_point = 0;
		cudaError_t cudaStatus = cudaSuccess;

		//Allocate device memory.
		cudaStatus = cudaMalloc((void**)&dev_point, count * sizeof(AffinePoint<N>));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		cudaStatus = cudaMalloc((void**)&dev_res, count * sizeof(AffinePoint<N>));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		cudaStatus = cudaMalloc((void**)&dev_scalar, count * sizeof(BigNum<N>));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		//Copy input arrays to device.
		cudaStatus = cudaMemcpy(dev_point, pointArray, count * sizeof(AffinePoint<N>), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		cudaStatus = cudaMemcpy(dev_scalar, scalarArray, count * sizeof(BigNum<N>), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		//Check if kernel launch returns non zero.
		if (launchScalarMultKernel(dev_res, dev_scalar, dev_point, curveType, count)) {
			goto Error;
		}

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "launchScalarMultKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		//Copy result to output array.
		cudaStatus = cudaMemcpy(resultArray, dev_res, count * sizeof(AffinePoint<N>), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy devicetohost launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

	Error:
		//Free pointers.
		cudaFree(dev_point);
		cudaFree(dev_scalar);
		cudaFree(dev_res);

		return cudaStatus;
	}
}

//Exposing for linkage only supported scalarMultiplyOnGpu.
template cudaError_t ScalarMultiply::scalarMultiplyOnGpu(AffinePoint<6> *resultArray, BigNum<6> *scalarArray,
	AffinePoint<6> *pointArray, unsigned int count, CurveType curveType);
template cudaError_t ScalarMultiply::scalarMultiplyOnGpu(AffinePoint<7> *resultArray, BigNum<7> *scalarArray,
	AffinePoint<7> *pointArray, unsigned int count, CurveType curveType);
template cudaError_t ScalarMultiply::scalarMultiplyOnGpu(AffinePoint<8> *resultArray, BigNum<8> *scalarArray,
	AffinePoint<8> *pointArray, unsigned int count, CurveType curveType);
template cudaError_t ScalarMultiply::scalarMultiplyOnGpu(AffinePoint<12> *resultArray, BigNum<12> *scalarArray,
	AffinePoint<12> *pointArray, unsigned int count, CurveType curveType);
template cudaError_t ScalarMultiply::scalarMultiplyOnGpu(AffinePoint<17> *resultArray, BigNum<17> *scalarArray,
	AffinePoint<17> *pointArray, unsigned int count, CurveType curveType);