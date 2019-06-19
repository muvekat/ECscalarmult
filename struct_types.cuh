#pragma once
#include "cuda.h"
#include "cuda_runtime.h"


//BigNumber representation.
template <size_t N>
struct BigNum {
	unsigned int value[N];

	__forceinline bool isZero() {
		for (int i = 0; i < N; ++i) {
			if (this->value[i] != 0) {
				return false;
			}
		}
		return true;
	}

	__device__ __forceinline__ bool isZeroDevice() {
		for (int i = 0; i < N; ++i) {
			if (this->value[i] != 0) {
				return false;
			}
		}
		return true;
	}
};


//Affine point representation.
//Point at infinity is x=0, y=0. 
template <size_t N>
struct AffinePoint {
	BigNum<N> x;
	BigNum<N> y;

	__forceinline bool isInfinity() {
		for (int i = 0; i < N; ++i) {
			if (this->x.value[i] != 0 || this->y.value[i] != 0) {
				return false;
			}
		}
		return true;
	}

	__device__ __forceinline__ bool isInfinityDevice() {
		for (int i = 0; i < N; ++i) {
			if (this->x.value[i] != 0 || this->y.value[i] != 0) {
				return false;
			}
		}
		return true;
	}
};

//Jacobian point representation.
//Point at infinity is (1:1:0).
template <size_t N>
struct JacobianPoint {
	BigNum<N> x;
	BigNum<N> y;
	BigNum<N> z;

	__forceinline bool isInfinity() {

		if (this->x.value[0] != 1 || this->y.value[0] != 1 || this->z.value[0] != 0) {
			return false;
		}

		for (int i = 1; i < N; ++i) {
			if (this->x.value[i] != 0 || this->y.value[i] != 0 || this->z.value[i] != 0) {
				return false;
			}
		}

		return true;
	}

	__device__ __forceinline__ bool isInfinityDevice() {

		if (this->x.value[0] != 1 || this->y.value[0] != 1 || this->z.value[0] != 0) {
			return false;
		}

		for (int i = 1; i < N; ++i) {
			if (this->x.value[i] != 0 || this->y.value[i] != 0 || this->z.value[i] != 0) {
				return false;
			}
		}

		return true;
	}
};

//Weierstrass curve representation.
template <size_t N>
struct WeierstrassCurve {
	BigNum<N> prime;
	BigNum<N> n;
	AffinePoint<N> G;
	BigNum<N> a;
	BigNum<N> b;
	BigNum<N>(*multiplyNumAndMod) (BigNum<N> first, BigNum<N> second);
	BigNum<N>(*multiplyNumIntAndMod) (BigNum<N> first, unsigned int second);
	BigNum<N>(*squareNumAndMod) (BigNum<N> first);
};

//Enumeration of supported curve types.
enum CurveType {
	secp192r1,
	secp224r1,
	secp256r1,
	secp384r1,
	secp521r1
};