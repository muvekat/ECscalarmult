#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "struct_types.cuh"

#include <stdio.h>

namespace PrimeFieldArithmetic {

	//Compute (first + second) in long arithmetic and set carryResult pointer value with carry bit.
	template <size_t N>
	__device__ BigNum<N> multiprecisionAddNum(unsigned int *carryResult,
		BigNum<N> first, BigNum<N> second)
	{
		BigNum<N> result;

		asm volatile ("add.cc.u32 %0, %1, %2;\n\t"
			: "=r"(result.value[0]) : "r"(first.value[0]), "r"(second.value[0]) : "memory");

		for (int ind = 1; ind < N - 1; ++ind) {
			asm volatile (
				"addc.cc.u32 %0, %1, %2;\n\t"
				: "=r"(result.value[ind]) : "r"(first.value[ind]), "r"(second.value[ind]) : "memory");
		}

		if (N > 2) {
			asm volatile (
				"addc.cc.u32 %0, %1, %2;\n\t"
				: "=r"(result.value[N - 1]) : "r"(first.value[N - 1]), "r"(second.value[N - 1]) : "memory");
		}

		asm volatile ("addc.u32 %0, 0, 0;\n\t" : "=r"(*carryResult) : : "memory");

		return result;
	}


	//Compute (first - second) in long arithmetic and set carryResult pointer value with carry bit.
	template <size_t N>
	__device__ BigNum<N> multiprecisionSubstractNum(unsigned int *carryResult,
		BigNum<N> first, BigNum<N> second)
	{
		BigNum<N> result;

		asm volatile (
			"sub.cc.u32 %0, %1, %2;\n\t"
			: "=r"(result.value[0]) : "r"(first.value[0]), "r"(second.value[0]) : "memory");

		for (int ind = 1; ind < N - 1; ++ind) {
			asm volatile (
				"subc.cc.u32 %0, %1, %2;\n\t"
				: "=r"(result.value[ind]) : "r"(first.value[ind]), "r"(second.value[ind]) : "memory");
		}

		if (N > 2) {
			asm volatile (
				"subc.cc.u32 %0, %1, %2;\n\t"
				: "=r"(result.value[N - 1]) : "r"(first.value[N - 1]), "r"(second.value[N - 1]) : "memory");
		}

		asm volatile ("subc.u32 %0, 0, 0;\n\t" : "=r"(*carryResult) : : "memory");
		//Warning: carryResult[idx] will get 0xffffffff if there was carry bit. Use only for checking if it's not 0.

		return result;
	}


	//Compute (first + second) in long arithmetic and disregard carry bit.
	template <size_t N>
	__device__ BigNum<N> multiprecisionAddNumWithoutCarry(BigNum<N> first, BigNum<N> second)
	{
		BigNum<N> result;

		asm volatile ("add.cc.u32 %0, %1, %2;\n\t"
			: "=r"(result.value[0]) : "r"(first.value[0]), "r"(second.value[0]) : "memory");

		for (int ind = 1; ind < N - 1; ++ind) {
			asm volatile (
				"addc.cc.u32 %0, %1, %2;\n\t"
				: "=r"(result.value[ind]) : "r"(first.value[ind]), "r"(second.value[ind]) : "memory");
		}

		if (N > 2) {
			asm volatile (
				"addc.u32 %0, %1, %2;\n\t"
				: "=r"(result.value[N - 1]) : "r"(first.value[N - 1]), "r"(second.value[N - 1]) : "memory");
		}

		return result;
	}


	//Compute (first - second) in long arithmetic and disregard carry bit.
	template <size_t N>
	__device__ BigNum<N> multiprecisionSubstractNumWithoutCarry(BigNum<N> first, BigNum<N> second)
	{
		BigNum<N> result;

		asm volatile (
			"sub.cc.u32 %0, %1, %2;\n\t"
			: "=r"(result.value[0]) : "r"(first.value[0]), "r"(second.value[0]) : "memory");

		for (int ind = 1; ind < N - 1; ++ind) {
			asm volatile (
				"subc.cc.u32 %0, %1, %2;\n\t"
				: "=r"(result.value[ind]) : "r"(first.value[ind]), "r"(second.value[ind]) : "memory");
		}

		if (N > 2) {
			asm volatile (
				"subc.u32 %0, %1, %2;\n\t"
				: "=r"(result.value[N - 1]) : "r"(first.value[N - 1]), "r"(second.value[N - 1]) : "memory");
		}

		return result;
	}


	//Check if (first >= second).
	template <size_t N>
	__device__ bool checkIfOverOrEqual(BigNum<N> first, BigNum<N> second)
	{
		for (int ind = N - 1; ind >= 0; --ind) { //Scan from highest word to lowest.
			if (first.value[ind] > second.value[ind] || (ind == 0 && first.value[ind] == second.value[ind])) {
				return true;
			}
			if (first.value[ind] < second.value[ind]) {
				return false;
			}
		}
		return false;
	}


	//Compute ((first + second) mod base).
	template <size_t N>
	__device__ BigNum<N> addNum(BigNum<N> first, BigNum<N> second,
		const BigNum<N> base)
	{
		BigNum<N> result;

		unsigned int carry;
		result = multiprecisionAddNum(&carry, first, second);
		if (carry || checkIfOverOrEqual(result, base)) {
			result = multiprecisionSubstractNumWithoutCarry(result, base);
		}

		return result;
	}


	//Compute ((first - second) mod base).
	template <size_t N>
	__device__ BigNum<N> substractNum(BigNum<N> first, BigNum<N> second,
		const BigNum<N> base)
	{
		BigNum<N> result;

		unsigned int carry;
		result = multiprecisionSubstractNum(&carry, first, second);
		if (carry) {
			result = multiprecisionAddNumWithoutCarry(result, base);
		}

		return result;
	}


	//Compute (first * second) in long arithmetic, without computing mod base.
	template <size_t N>
	__device__ BigNum<2 * N> multiplyNum(BigNum<N> first, BigNum<N> second)
	{
		BigNum<2 * N> result;
		asm volatile ("{\n\t");
		int iInd, jInd, kInd;
		int intN = (int)N;
		unsigned int tmp1 = 0;
		unsigned int tmp2 = 0;
		unsigned int tmp3 = 0;
		asm volatile (".reg .u32 t1, t2;\n\t" : : : "memory"); // Allocate registers.

		//For each element of {(i,j)} | i+j=k, 0 <= i, j <= t-1}...
		for (kInd = 0; kInd < 2 * N - 1; ++kInd) {
			for (jInd = max(0, kInd - (intN - 1)); jInd < N && jInd <= kInd; ++jInd) {
				iInd = kInd - jInd;
				asm volatile (
					"mul.hi.u32 t1, %3, %4;\n\t"
					"mul.lo.u32 t2, %3, %4;\n\t"
					"add.cc.u32 %0, %0, t2;\n\t"
					"addc.cc.u32 %1, %1, t1;\n\t"
					"addc.u32 %2, %2, 0;\n\t"
					: "+r"(tmp1), "+r"(tmp2), "+r"(tmp3) : "r"(first.value[iInd]), "r"(second.value[jInd]) : "memory");
			}
			result.value[kInd] = tmp1;
			tmp1 = tmp2; tmp2 = tmp3; tmp3 = 0;
		}
		result.value[2 * N - 1] = tmp1;

		asm volatile ("}\n\t");
		return result;
	}


	//Compute (first * second) in long arithmetic, without computing mod base.
	//Used for cases where secon is a small number which fits in a single unsigned int.
	template <size_t N>
	__device__ BigNum<2 * N> multiplyNumInt(BigNum<N> first, unsigned int second)
	{
		asm volatile ("{\n\t");
		BigNum<2 * N> result = {};
		asm volatile (".reg .u32 t1, t2;\n\t" : : : "memory"); // Allocate registers.

		for (int kInd = 0; kInd < N; ++kInd) {
			asm volatile (
				"mul.hi.u32 t1, %3, %4;\n\t"
				"mul.lo.u32 t2, %3, %4;\n\t"
				"add.cc.u32 %0, %0, t2;\n\t"
				"addc.cc.u32 %1, %1, t1;\n\t"
				"addc.u32 %2, %2, 0;\n\t"
				: "+r"(result.value[kInd]), "+r"(result.value[kInd+1]), "+r"(result.value[kInd+2]) : "r"(first.value[kInd]), "r"(second) : "memory");
		}

		asm volatile ("}\n\t");
		return result;
	}


	//Compute (number ^ 2) in long arithmetic, without computing mod base.
	template <size_t N>
	__device__ BigNum<2 * N> squareNum(BigNum<N> number)
	{
		asm volatile ("{\n\t");
		BigNum<2 * N> result;
		int iInd, jInd, kInd;
		unsigned int tmp1 = 0;
		unsigned int tmp2 = 0;
		unsigned int tmp3 = 0;
		asm volatile (".reg .u32 t1, t2;\n\t" : : : "memory"); // Allocate registers.

		//For each element of {(i,j)} | i+j=k, 0 <= i <= j <= t-1}...
		for (kInd = 0; kInd < 2 * N - 1; ++kInd) {
			for (jInd = (kInd >> 1) + (kInd & 1); jInd < N && jInd <= kInd; ++jInd) { //jInd = kInd / 2 + kInd % 2
				iInd = kInd - jInd;
				asm volatile ("mul.hi.u32 t1, %0, %1;\n\t"
					"mul.lo.u32 t2, %0, %1;\n\t"
					: : "r"(number.value[iInd]), "r"(number.value[jInd]) : "memory");
				if (iInd < jInd) { //multiply by 2
					asm volatile ("add.cc.u32 t2, t2, t2;\n\t"
						"addc.cc.u32 t1, t1, t1;\n\t"
						"addc.u32 %0, %0, 0;\n\t"
						: "+r"(tmp3) : : "memory");
				}
				asm volatile ("add.cc.u32 %0, %0, t2;\n\t"
					"addc.cc.u32 %1, %1, t1;\n\t"
					"addc.u32 %2, %2, 0;\n\t"
					: "+r"(tmp1), "+r"(tmp2), "+r"(tmp3) : : "memory");
			}
			result.value[kInd] = tmp1;
			tmp1 = tmp2; tmp2 = tmp3; tmp3 = 0;
		}
		result.value[2 * N - 1] = tmp1;
		asm volatile("}\n\t");
		return result;
	}


	//Checks if (number != 1).
	template <size_t N>
	__device__ bool isNotOne(BigNum<N> number)
	{
		if (number.value[0] != 1) {
			return true;
		}

		for (int ind = 1; ind < N; ++ind) {
			if (number.value[ind] != 0) {
				return true;
			}
		}

		return false;
	}


	//Checks if (number != 0).
	template <size_t N>
	__device__ bool isNotZero(BigNum<N> number)
	{
		for (int ind = 0; ind < N; ++ind) {
			if (number.value[ind] != 0) {
				return true;
			}
		}

		return false;
	}


	//Checks if (number is even).
	template <size_t N>
	__device__ bool isEven(BigNum<N> number)
	{
		return !(number.value[0] & 1);
	}


	//Computes a value which is a result of shifting number right by 1 and returns it.
	template <size_t N>
	__device__ BigNum<N> shiftRight(BigNum<N> number)
	{
		BigNum<N> result;
		unsigned int carry1 = 0;
		unsigned int carry2 = 0;

		for (int ind = N - 1; ind > 0; --ind) {
			carry2 = (number.value[ind] & 1) << 31;
			result.value[ind] = number.value[ind] >> 1;
			result.value[ind] |= carry1;
			carry1 = carry2;
		}
		result.value[0] = number.value[0] >> 1;
		result.value[0] |= carry1;

		return result;
	}

	//Computes a value which is a result of adding addValue to number
	// and shifting resulting number right by 1 and returns it.
	template <size_t N>
	__device__ BigNum<N> addAndShiftRight(BigNum<N> number, BigNum<N> addValue)
	{
		BigNum<N> result;
		unsigned int carry;
		result = multiprecisionAddNum<N>(&carry, number, addValue);
		result = shiftRight(result);

		if (carry) {
			carry <<= 31;
			result.value[N - 1] |= carry;
		}

		return result;
	}


	//Computes ((number ^ -1) mod base)
	template <size_t N>
	__device__ BigNum<N> inverseNum(BigNum<N> number, const BigNum<N> base) {
		BigNum<N> u, v, x1, x2;

		//Initialization .
		u = number;
		v = base;
		x1 = { 1 };
		x2 = { 0 };

		while (isNotOne(u) && isNotOne(v)) { //u !=1 && v != 1

			while (isEven(u) && isNotZero(u)) { //u is even
				u = shiftRight(u);

				if (isEven(x1) && isNotZero(x1)) { //x1 is even
					x1 = shiftRight(x1);
				}
				else {
					x1 = addAndShiftRight(x1, base);
				}
			}

			while (isEven(v) && isNotZero(v)) { //v is even
				v = shiftRight(v);

				if (isEven(x2) && isNotZero(x2)) { //x2 is even
					x2 = shiftRight(x2);
				}
				else {
					x2 = addAndShiftRight(x2, base);
				}
			}

			if (checkIfOverOrEqual(u, v)) { //u >= v
				u = multiprecisionSubstractNumWithoutCarry(u, v);
				x1 = substractNum(x1, x2, base);
			}
			else {
				v = multiprecisionSubstractNumWithoutCarry(v, u);
				x2 = substractNum(x2, x1, base);
			}
		}

		if (u.value[0] == 1) {
			return x1;
		}
		else {
			return x2;
		}
	}

	//Computes ((number / divisor) mod base)
	template <size_t N>
	__device__ BigNum<N> divide(BigNum<N> number, BigNum<N> divisor, const BigNum<N> base) {
		BigNum<N> u, v, x1, x2;

		//Initialization .
		u = divisor;
		v = base;
		x1 = number;
		x2 = { 0 };

		while (isNotOne(u) && isNotOne(v)) { //u !=1 && v != 1

			while (isEven(u) && isNotZero(u)) { //u is even
				u = shiftRight(u);

				if (isEven(x1) && isNotZero(x1)) { //x1 is even
					x1 = shiftRight(x1);
				}
				else {
					x1 = addAndShiftRight(x1, base);
				}
			}

			while (isEven(v) && isNotZero(v)) { //v is even
				v = shiftRight(v);

				if (isEven(x2) && isNotZero(x2)) { //x2 is even
					x2 = shiftRight(x2);
				}
				else {
					x2 = addAndShiftRight(x2, base);
				}
			}

			if (checkIfOverOrEqual(u, v)) { //u >= v
				u = multiprecisionSubstractNumWithoutCarry(u, v);
				x1 = substractNum(x1, x2, base);
			}
			else {
				v = multiprecisionSubstractNumWithoutCarry(v, u);
				x2 = substractNum(x2, x1, base);
			}
		}

		if (u.value[0] == 1) {
			return x1;
		}
		else {
			return x2;
		}
	}

}
