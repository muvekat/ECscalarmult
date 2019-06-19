#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "prime_field_arithmetic.cuh"
#include "struct_types.cuh"

#include <stdio.h>

namespace PrimeCurveArithmetic {

	namespace pfa = PrimeFieldArithmetic;



	template <size_t N>
	__device__ JacobianPoint<N> affineToJacobian(AffinePoint<N> p)
	{
		JacobianPoint<N> result;
		result.z = BigNum<N>{ 1 };

		result.x = p.x;
		result.y = p.y;

		return result;
	}

	template <size_t N>
	__device__ JacobianPoint<N> jacobianScaleZ(JacobianPoint<N> p, const WeierstrassCurve<N> curve)
	{
		JacobianPoint<N> result;
		BigNum<N> invZ, invZsq;

		invZ = pfa::inverseNum(p.z, curve->prime);			// 1/z
		invZsq = curve->squareNumAndMod(invZ);				// 1/z^2

		result.x = curve->multiplyNumAndMod(p.x, invZsq);	// x = x/z^2
		invZ = curve->multiplyNumAndMod(invZ, invZsq);		// 1/z^3
		result.y = curve->multiplyNumAndMod(p.y, invZ);		// y = y/z^3
		result.z = BigNum<N>{ 1 };						// z = 1

		return result;
	}

	template <size_t N>
	__device__ AffinePoint<N> jacobianToAffine(JacobianPoint<N> p, const WeierstrassCurve<N> *curve)
	{
		AffinePoint<N> result;
		BigNum<N> invZ, invZsq;

		invZ = pfa::inverseNum(p.z, curve->prime);			// 1/z
		invZsq = curve->squareNumAndMod(invZ);				// 1/z^2

		result.x = curve->multiplyNumAndMod(p.x, invZsq);	// x = x/z^2
		invZ = curve->multiplyNumAndMod(invZ, invZsq);		// 1/z^3
		result.y = curve->multiplyNumAndMod(p.y, invZ);		// y = y/z^3

		return result;
	}

	template <size_t N>
	__device__ __noinline__ bool shouldComputeDouble(JacobianPoint<N> p)
	{
		if (p.x.value[0] != 0 || p.y.value[0] != 1 || p.z.value[0] != 0) {
			return false;
		}

		for (int i = 1; i < N; ++i) {
			if (p.x.value[i] != 0 || p.y.value[i] != 0 || p.z.value[i] != 0) {
				return false;
			}
		}

		return true;
	}

	template <size_t N>
	__device__ JacobianPoint<N> addPoint(JacobianPoint<N> first, AffinePoint<N> second,
		const WeierstrassCurve<N> *curve)
	{
		if (second.isInfinityDevice()) {
			return first;
		}
		if (first.isInfinityDevice()) {
			return affineToJacobian(second);
		}

		JacobianPoint<N> result;
		BigNum<N> t1, t2, t3, t4;

		t1 = curve->squareNumAndMod(first.z);						//T1 = Z1^2
		t2 = curve->multiplyNumAndMod(t1, first.z);					//T2 = T1*Z1
		t1 = curve->multiplyNumAndMod(t1, second.x);				//T1 = T1*X2
		t2 = curve->multiplyNumAndMod(t2, second.y);				//T2 = T2*Y2
		t1 = pfa::substractNum(t1, first.x, curve->prime);			//T1 = T1-X1
		t2 = pfa::substractNum(t2, first.y, curve->prime);			//T2 = T2-Y1

		//Check if we got double point or point at infinity.
		if (t1.isZeroDevice()) {
			if (t2.isZeroDevice()) {
				return JacobianPoint<N>{ {0}, { 1 }, { 0 } }; // Dummy point used to check if we should compute double.
			}
				return JacobianPoint<N>{ {1},{1},{0} }; //Point at infinity.
			
		}

		result.z = curve->multiplyNumAndMod(first.z, t1);			//Z3 = Z1*T1
		t3 = curve->squareNumAndMod(t1);							//T3 = T1^2
		t4 = curve->multiplyNumAndMod(t3, t1);						//T4 = T3 * T1
		t3 = curve->multiplyNumAndMod(t3, first.x);					//T3 = T3 * X1
		t1 = pfa::addNum(t3, t3, curve->prime);						//T1 = 2 * T3
		result.x = curve->squareNumAndMod(t2);						//X3 = T2^2
		result.x = pfa::substractNum(result.x, t1, curve->prime);	//X3 = X3 - T1
		result.x = pfa::substractNum(result.x, t4, curve->prime);	//X3 = X3 - T4
		t3 = pfa::substractNum(t3, result.x, curve->prime);			//T3 = T3 - X3
		t3 = curve->multiplyNumAndMod(t3, t2);						//T3 = T3 * T2
		t4 = curve->multiplyNumAndMod(t4, first.y);					//T4 = T4 * Y1
		result.y = pfa::substractNum(t3, t4, curve->prime);			//Y3 = T3 - T4

		return result;
	}

	template <size_t N>
	__device__ JacobianPoint<N> doublePoint(JacobianPoint<N> point, const WeierstrassCurve<N> *curve)
	{
		if (point.isInfinityDevice()) {
			return point;
		}

		JacobianPoint<N> result;
		BigNum<N> t1, t2, t3;

		t1 = curve->squareNumAndMod(point.z);						//T1 = Z1^2
		t2 = pfa::substractNum(point.x, t1, curve->prime);			//T2 = X1 - T1
		t1 = pfa::addNum(point.x, t1, curve->prime);				//T1 = X1 + T1
		t2 = curve->multiplyNumAndMod(t2, t1);						//T2 = T2 * T1
		t2 = curve->multiplyNumIntAndMod(t2, 3);					//T2 = 3 * T2
		result.y = pfa::addNum(point.y, point.y, curve->prime);		//Y3 = 2 * Y1
		result.z = curve->multiplyNumAndMod(result.y, point.z);		//Z3 = Y3 * Z1
		result.y = curve->squareNumAndMod(result.y);				//Y3 = Y3^2
		t3 = curve->multiplyNumAndMod(result.y, point.x);			//T3 = Y3 * X1
		result.y = curve->squareNumAndMod(result.y);				//Y3 = Y3^2
		result.y = pfa::divide(result.y, { 2 }, curve->prime);		//Y3 = half * Y3
		result.x = curve->squareNumAndMod(t2);						//X3 = T2^2
		t1 = pfa::addNum(t3, t3, curve->prime);						//T1 = 2 * T3
		result.x = pfa::substractNum(result.x, t1, curve->prime);	//X3 = X3 - T1
		t1 = pfa::substractNum(t3, result.x, curve->prime);			//T1 = T3 - X3
		t1 = curve->multiplyNumAndMod(t1, t2);						//T1 = T1 * T2
		result.y = pfa::substractNum(t1, result.y, curve->prime);	//Y3 = T1 - Y3

		return result;
	}

	template <size_t N>
	__device__ AffinePoint<N> scalarMultBinary(BigNum<N> scalar, AffinePoint<N> point,
		const WeierstrassCurve<N> *curve)
	{
		AffinePoint<N> result;
		int iInd, kInd;
		JacobianPoint<N> qPoint;

		qPoint = affineToJacobian(point);

		//Find the start index.
		for (kInd = N - 1; kInd >= 0; kInd--) {
			asm volatile ("bfind.u32 %0, %1;\n\t" : "=r"(iInd) : "r"(scalar.value[kInd]));

			if (iInd != 0xffffffff) {
				iInd--;
				break;
			}

		}

		//Main loop.
		for (; kInd >= 0; kInd--) {
			for (; iInd >= 0; iInd--) {
				qPoint = doublePoint(qPoint, curve);
				if (scalar.value[kInd] & (1 << iInd)) {
					qPoint = addPoint(qPoint, point, curve);
					if (shouldComputeDouble(qPoint)) {
						//Trick to compute double point with shorter generated code length to reduce exec time.
						scalar.value[kInd] ^= (1 << iInd);
						qPoint = affineToJacobian(point);
						iInd++;
					}
				}
			}
			iInd = 31;
		}

		result = jacobianToAffine(qPoint, curve);

		return result;
	}
}