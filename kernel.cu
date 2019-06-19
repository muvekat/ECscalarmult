
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "struct_types.cuh"
#include "scalar_multiplication.cuh"

#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <ctime>

template <size_t N>
void generateScalarArray(BigNum<N> *arr, int count) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<unsigned int> dis(0, UINT32_MAX);
	for (int i = 0; i < count; i++) {
		for (int j = 0; j < N; j++) {
			arr[i].value[j] = dis(gen);
		}
	}
}

template <size_t N>
void generatePointArray(AffinePoint<N> *arr, int count) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<unsigned int> dis(0, UINT32_MAX);
	for (int i = 0; i < count; i++) {
		for (int j = 0; j < N; j++) {
			arr[i].x.value[j] = dis(gen);
			arr[i].y.value[j] = dis(gen);
		}
	}
}

template <size_t N>
void printScalarFirstElement(BigNum<N> *arr) {
	for (int j = 0; j < N; j++) {
		printf("%x ", arr[0].value[j]);
	}
	printf("\n");
}

template <size_t N>
void printPointFirstElement(AffinePoint<N> *arr) {
	printf("x ");
	for (int j = 0; j < N; j++) {
		printf("%x ", arr[0].x.value[j]);
	}
	printf("y ");
	for (int j = 0; j < N; j++) {
		printf("%x ", arr[0].y.value[j]);
	}
	printf("\n");
}

int main()
{
	//Set launch device.
	cudaError_t cudaStat = cudaSetDevice(0);
	if (cudaStat != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	cudaDeviceSynchronize();

	/*test block*/
	{

		const int num_elems = 65536;
		AffinePoint<6> *result;
		BigNum<6> *scalar;
		AffinePoint<6> *point;
		point = (AffinePoint<6>*) malloc(num_elems * sizeof(AffinePoint<6>));
		scalar = (BigNum<6>*) malloc(num_elems * sizeof(BigNum<6>));
		result = (AffinePoint<6>*) malloc(num_elems * sizeof(AffinePoint<6>));

		generateScalarArray(scalar, num_elems); 

		for (int i = 0; i < num_elems; i++) {
			point[i] = { {0x82ff1012, 0xf4ff0afd, 0x43a18800, 0x7cbf20eb, 0xb03090f6, 0x188da80e},
			{ 0x1e794811, 0x73f977a1, 0x6b24cdd5, 0x631011ed, 0xffc8da78, 0x7192b95} };
		}

		printf("scalar mult secp192r1:\n");
		printScalarFirstElement(scalar);
		printPointFirstElement(point);


		auto exectime = std::chrono::high_resolution_clock::now();
		ScalarMultiply::scalarMultiplyOnGpu(result, scalar, point, num_elems, CurveType::secp192r1);
		auto time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - exectime).count();
		printf("exec time %lf ms\n", time);

		printf("result:\n");
		printPointFirstElement(result);


		free(point); free(scalar); free(result);
	}
	{
		const int num_elems = 65536;
		AffinePoint<7> *result;
		BigNum<7> *scalar;
		AffinePoint<7> *point;
		point = (AffinePoint<7>*) malloc(num_elems * sizeof(AffinePoint<7>));
		scalar = (BigNum<7>*) malloc(num_elems * sizeof(BigNum<77>));
		result = (AffinePoint<7>*) malloc(num_elems * sizeof(AffinePoint<7>));

		generateScalarArray(scalar, num_elems);

		for (int i = 0; i < num_elems; i++) {
			point[i] = { { 0x115C1D21, 0x343280D6, 0x56C21122, 0x4A03C1D3, 0x321390B9, 0x6BB4BF7F, 0xB70E0CBD },
			{ 0x85007E34, 0x44D58199, 0x5A074764, 0xCD4375A0, 0x4C22DFE6, 0xB5F723FB, 0xBD376388 } };
		}

		printf("scalar mult secp224r1:\n");
		printScalarFirstElement(scalar);
		printPointFirstElement(point);


		auto exectime = std::chrono::high_resolution_clock::now();
		ScalarMultiply::scalarMultiplyOnGpu(result, scalar, point, num_elems, CurveType::secp224r1);
		auto time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - exectime).count();
		printf("exec time %lf ms\n", time);

		printf("result:\n");
		printPointFirstElement(result);

		free(point); free(scalar); free(result);
	}
	{
		const int num_elems = 65536;
		AffinePoint<8> *result;
		BigNum<8> *scalar;
		AffinePoint<8> *point;
		point = (AffinePoint<8>*) malloc(num_elems * sizeof(AffinePoint<8>));
		scalar = (BigNum<8>*) malloc(num_elems * sizeof(BigNum<8>));
		result = (AffinePoint<8>*) malloc(num_elems * sizeof(AffinePoint<8>));

		generateScalarArray(scalar, num_elems);

		for (int i = 0; i < num_elems; i++) {
			point[i] = { { 0xD898C296, 0xF4A13945, 0x2DEB33A0, 0x77037D81, 0x63A440F2, 0xF8BCE6E5, 0xE12C4247, 0x6B17D1F2 },
			{ 0x37BF51F5, 0xCBB64068, 0x6B315ECE, 0x2BCE3357,  0x7C0F9E16, 0x8EE7EB4A, 0xFE1A7F9B, 0x4FE342E2 } };
		}

		printf("scalar mult secp256r1:\n");
		printScalarFirstElement(scalar);
		printPointFirstElement(point);


		auto exectime = std::chrono::high_resolution_clock::now();
		ScalarMultiply::scalarMultiplyOnGpu(result, scalar, point, num_elems, CurveType::secp256r1);
		auto time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - exectime).count();
		printf("exec time %lf ms\n", time);

		printf("result:\n");
		printPointFirstElement(result);

		free(point); free(scalar); free(result);
	}
	{
		const int num_elems = 65536;
		AffinePoint<12> *result;
		BigNum<12> *scalar;
		AffinePoint<12> *point;
		point = (AffinePoint<12>*) malloc(num_elems * sizeof(AffinePoint<12>));
		scalar = (BigNum<12>*) malloc(num_elems * sizeof(BigNum<12>));
		result = (AffinePoint<12>*) malloc(num_elems * sizeof(AffinePoint<12>));

		generateScalarArray(scalar, num_elems);

		for (int i = 0; i < num_elems; i++) {
			point[i] = { { 0x72760AB7, 0x3A545E38, 0xBF55296C, 0x5502F25D, 0x82542A38, 0x59F741E0, 0x8BA79B98, 0x6E1D3B62, 0xF320AD74, 0x8EB1C71E, 0xBE8B0537, 0xAA87CA22 },
			{ 0x90EA0E5F, 0x7A431D7C, 0x1D7E819D, 0x0A60B1CE, 0xB5F0B8C0, 0xE9DA3113, 0x289A147C, 0xF8F41DBD, 0x9292DC29, 0x5D9E98BF, 0x96262C6F, 0x3617DE4A } };
		}

		printf("scalar mult secp384r1:\n");
		printScalarFirstElement(scalar);
		printPointFirstElement(point);


		auto exectime = std::chrono::high_resolution_clock::now();
		ScalarMultiply::scalarMultiplyOnGpu(result, scalar, point, num_elems, CurveType::secp384r1);
		auto time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - exectime).count();
		printf("exec time %lf ms\n", time);

		printf("result:\n");
		printPointFirstElement(point);

		free(point); free(scalar); free(result);
	}
	{
		const int num_elems = 65536;
		AffinePoint<17> *result;
		BigNum<17> *scalar;
		AffinePoint<17> *point;
		point = (AffinePoint<17>*) malloc(num_elems * sizeof(AffinePoint<17>));
		scalar = (BigNum<17>*) malloc(num_elems * sizeof(BigNum<17>));
		result = (AffinePoint<17>*) malloc(num_elems * sizeof(AffinePoint<17>));

		generateScalarArray(scalar, num_elems);

		for (int i = 0; i < num_elems; i++) {
			point[i] = { { 0xC2E5BD66, 0xF97E7E31, 0x856A429B, 0x3348B3C1, 0xA2FFA8DE, 0xFE1DC127, 0xEFE75928, 0xA14B5E77, 0x6B4D3DBA, 0xF828AF60, 0x053FB521, 0x9C648139,
		0x2395B442, 0x9E3ECB66, 0x0404E9CD, 0x858E06B7, 0xC6 },
			{ 0x9FD16650, 0x88BE9476, 0xA272C240, 0x353C7086, 0x3FAD0761, 0xC550B901, 0x5EF42640, 0x97EE7299, 0x273E662C, 0x17AFBD17, 0x579B4468, 0x98F54449,
			0x2C7D1BD9, 0x5C8A5FB4, 0x9A3BC004, 0x39296A78, 0x118 } };
		}

		printf("scalar mult secp521r1:\n");
		printScalarFirstElement(scalar);
		printPointFirstElement(point);


		auto exectime = std::chrono::high_resolution_clock::now();
		ScalarMultiply::scalarMultiplyOnGpu(result, scalar, point, num_elems, CurveType::secp521r1);
		auto time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - exectime).count();
		printf("exec time %lf ms\n", time);

		printf("result:\n");
		printPointFirstElement(result);

		free(point); free(scalar); free(result);
	}
	/*end check arithm block*/

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}