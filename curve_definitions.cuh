#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "prime_field_arithmetic.cuh"
#include "prime_curve_arithmetic.cuh"
#include "struct_types.cuh"

namespace Curve {

	namespace pfa = PrimeFieldArithmetic;
	namespace pca = PrimeCurveArithmetic;

	__constant__ static const BigNum<6> P192base = { 0xffffffff, 0xffffffff, 0xfffffffe, 0xffffffff, 0xffffffff, 0xffffffff };
	__device__ BigNum<6> fastRecuctionP192(BigNum<12> number) {
		BigNum<6> s1, s2, s3, s4, tmp1, tmp2, result;

		//Initialization of S values
		//s1=(c2,c1,c0) of 64 bit values
		s1.value[0] = number.value[0];
		s1.value[1] = number.value[1];
		s1.value[2] = number.value[2];
		s1.value[3] = number.value[3];
		s1.value[4] = number.value[4];
		s1.value[5] = number.value[5];
		//s2=(0,c3,c3) of 64 bit values
		s2.value[0] = number.value[6];
		s2.value[1] = number.value[7];
		s2.value[2] = number.value[6];
		s2.value[3] = number.value[7];
		s2.value[4] = 0;
		s2.value[5] = 0;
		//s3=(c4,c4,0) of 64 bit values
		s3.value[0] = 0;
		s3.value[1] = 0;
		s3.value[2] = number.value[8];
		s3.value[3] = number.value[9];
		s3.value[4] = number.value[8];
		s3.value[5] = number.value[9];
		//s4=(c5,c5,c5) of 64 bit values
		s4.value[0] = number.value[10];
		s4.value[1] = number.value[11];
		s4.value[2] = number.value[10];
		s4.value[3] = number.value[11];
		s4.value[4] = number.value[10];
		s4.value[5] = number.value[11];

		//result = s1 + s2 + s3 + s4
		tmp1 = pfa::addNum(s1, s2, P192base);
		tmp2 = pfa::addNum(s3, s4, P192base);
		result = pfa::addNum(tmp1, tmp2, P192base);

		return result;
	}

	__constant__ static const BigNum<7> P224base = { 1, 0, 0, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff };
	__device__ BigNum<7> fastRecuctionP224(BigNum<14> number) {
		BigNum<7> s1, s2, s3, s4, s5, tmp1, tmp2, result;

		//Initialization of S values
		//s1=(c6, c5, c4, c3, c2, c1, c0)
		s1.value[0] = number.value[0];
		s1.value[1] = number.value[1];
		s1.value[2] = number.value[2];
		s1.value[3] = number.value[3];
		s1.value[4] = number.value[4];
		s1.value[5] = number.value[5];
		s1.value[6] = number.value[6];
		//s2=(c10, c9, c8, c7, 0, 0, 0)
		s2.value[0] = 0;
		s2.value[1] = 0;
		s2.value[2] = 0;
		s2.value[3] = number.value[7];
		s2.value[4] = number.value[8];
		s2.value[5] = number.value[9];
		s2.value[6] = number.value[10];
		//s3=(0, c13, c12, c11, 0, 0, 0)
		s3.value[0] = 0;
		s3.value[1] = 0;
		s3.value[2] = 0;
		s3.value[3] = number.value[11];
		s3.value[4] = number.value[12];
		s3.value[5] = number.value[13];
		s3.value[6] = 0;
		//s4=(c13, c12, c11, c10, c9, c8, c7)
		s4.value[0] = number.value[7];
		s4.value[1] = number.value[8];
		s4.value[2] = number.value[9];
		s4.value[3] = number.value[10];
		s4.value[4] = number.value[11];
		s4.value[5] = number.value[12];
		s4.value[6] = number.value[13];
		//s5=(0, 0, 0, 0, c13, c12, c11).
		s5.value[0] = number.value[11];
		s5.value[1] = number.value[12];
		s5.value[2] = number.value[13];
		s5.value[3] = 0;
		s5.value[4] = 0;
		s5.value[5] = 0;
		s5.value[6] = 0;

		//result = s1 + s2 + s3 − s4 −s5
		tmp1 = pfa::addNum(s1, s2, P224base);				//s1+s2
		tmp2 = pfa::addNum(s4, s5, P224base);				//-s4-s5
		tmp1 = pfa::addNum(tmp1, s3, P224base);				//s1+s2+s3
		result = pfa::substractNum(tmp1, tmp2, P224base);	//s1 + s2 + s3 − s4 −s5

		return result;
	}

	__constant__ static const BigNum<8> P256base = { 0xffffffff, 0xffffffff, 0xffffffff, 0, 0, 0, 0x1, 0xffffffff };
	__device__ BigNum<8> fastRecuctionP256(BigNum<16> number) {
		BigNum<8> s1, s2, s3, s4, s5, s6, s7, s8, s9,
			tmp1, tmp2, tmp3, tmp4, result;


		//Initialization of S values
		//s1=(c7,c6,c5,c4,c3,c2,c1,c0) 
		s1.value[0] = number.value[0];
		s1.value[1] = number.value[1];
		s1.value[2] = number.value[2];
		s1.value[3] = number.value[3];
		s1.value[4] = number.value[4];
		s1.value[5] = number.value[5];
		s1.value[6] = number.value[6];
		s1.value[7] = number.value[7];
		//s2=(c15,c14,c13,c12,c11,0,0,0)
		s2.value[0] = 0;
		s2.value[1] = 0;
		s2.value[2] = 0;
		s2.value[3] = number.value[11];
		s2.value[4] = number.value[12];
		s2.value[5] = number.value[13];
		s2.value[6] = number.value[14];
		s2.value[7] = number.value[15];
		//s3=(0,c15,c14,c13,c12,0,0,0)
		s3.value[0] = 0;
		s3.value[1] = 0;
		s3.value[2] = 0;
		s3.value[3] = number.value[12];
		s3.value[4] = number.value[13];
		s3.value[5] = number.value[14];
		s3.value[6] = number.value[15];
		s3.value[7] = 0;
		//s4=(c15,c14,0,0,0,c10,c9,c8)
		s4.value[0] = number.value[8];
		s4.value[1] = number.value[9];
		s4.value[2] = number.value[10];
		s4.value[3] = 0;
		s4.value[4] = 0;
		s4.value[5] = 0;
		s4.value[6] = number.value[14];
		s4.value[7] = number.value[15];
		//s5=(c8,c13,c15,c14,c13,c11,c10,c9)
		s5.value[0] = number.value[9];
		s5.value[1] = number.value[10];
		s5.value[2] = number.value[11];
		s5.value[3] = number.value[13];
		s5.value[4] = number.value[14];
		s5.value[5] = number.value[15];
		s5.value[6] = number.value[13];
		s5.value[7] = number.value[8];
		//s6=(c10, c8,0,0,0, c13, c12, c11)
		s6.value[0] = number.value[11];
		s6.value[1] = number.value[12];
		s6.value[2] = number.value[13];
		s6.value[3] = 0;
		s6.value[4] = 0;
		s6.value[5] = 0;
		s6.value[6] = number.value[8];
		s6.value[7] = number.value[10];
		//s7=(c11, c9,0,0, c15, c14, c13, c12)
		s7.value[0] = number.value[12];
		s7.value[1] = number.value[13];
		s7.value[2] = number.value[14];
		s7.value[3] = number.value[15];
		s7.value[4] = 0;
		s7.value[5] = 0;
		s7.value[6] = number.value[9];
		s7.value[7] = number.value[11];
		//s8=(c12,0, c10, c9, c8, c15, c14, c13)
		s8.value[0] = number.value[13];
		s8.value[1] = number.value[14];
		s8.value[2] = number.value[15];
		s8.value[3] = number.value[8];
		s8.value[4] = number.value[9];
		s8.value[5] = number.value[10];
		s8.value[6] = 0;
		s8.value[7] = number.value[12];
		//s9=(c13,0, c11, c10, c9,0, c15, c14)
		s9.value[0] = number.value[14];
		s9.value[1] = number.value[15];
		s9.value[2] = 0;
		s9.value[3] = number.value[9];
		s9.value[4] = number.value[10];
		s9.value[5] = number.value[11];
		s9.value[6] = 0;
		s9.value[7] = number.value[13];

		//result = s1 + 2*s2 + 2*s3 + s4 + s5 − s6 − s7 − s8 − s9
		tmp1 = pfa::addNum(s2, s3, P256base);				//s2+s3
		tmp2 = pfa::addNum(s1, s4, P256base);				//s1+s4
		tmp3 = pfa::addNum(s6, s7, P256base);				//-s6-s7
		tmp4 = pfa::addNum(s8, s9, P256base);				//-s8-s9
		tmp1 = pfa::addNum(tmp1, tmp1, P256base);			//2*s2+2*s3. Using add since PTX doesnt have shift with carry.
		tmp2 = pfa::addNum(tmp2, s5, P256base);				//s1+s4+s5
		tmp3 = pfa::addNum(tmp3, tmp4, P256base);			//-s6-s7-s8-s9
		tmp1 = pfa::addNum(tmp1, tmp2, P256base);			//s1+2*s2+2*s3+s4+s5
		result = pfa::substractNum(tmp1, tmp3, P256base);	//s1 + 2*s2 + 2*s3 + s4 + s5 − s6 − s7 − s8 − s9

		return result;
	}

	__constant__ static const BigNum<12> P384base = { 0xffffffff, 0, 0, 0xffffffff, 0xfffffffe, 0xffffffff,
		0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff };
	__device__ BigNum<12> fastRecuctionP384(BigNum<24> number) {
		BigNum<12> s1, s2, s3, s4, s5, s6, s7, s8, s9, s10,
			tmp1, tmp2, tmp3, tmp4, tmp5, result;

		//Initialization of S values
		//s1=(c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0), 
		s1.value[0] = number.value[0];
		s1.value[1] = number.value[1];
		s1.value[2] = number.value[2];
		s1.value[3] = number.value[3];
		s1.value[4] = number.value[4];
		s1.value[5] = number.value[5];
		s1.value[6] = number.value[6];
		s1.value[7] = number.value[7];
		s1.value[8] = number.value[8];
		s1.value[9] = number.value[9];
		s1.value[10] = number.value[10];
		s1.value[11] = number.value[11];
		//s2=(0,0,0,0,0, c23, c22, c21,0,0,0,0)
		s2.value[0] = 0;
		s2.value[1] = 0;
		s2.value[2] = 0;
		s2.value[3] = 0;
		s2.value[4] = number.value[21];
		s2.value[5] = number.value[22];
		s2.value[6] = number.value[23];
		s2.value[7] = 0;
		s2.value[8] = 0;
		s2.value[9] = 0;
		s2.value[10] = 0;
		s2.value[11] = 0;
		//s3=(c23, c22, c21, c20, c19, c18, c17, c16, c15, c14, c13, c12)
		s3.value[0] = number.value[12];
		s3.value[1] = number.value[13];
		s3.value[2] = number.value[14];
		s3.value[3] = number.value[15];
		s3.value[4] = number.value[16];
		s3.value[5] = number.value[17];
		s3.value[6] = number.value[18];
		s3.value[7] = number.value[19];
		s3.value[8] = number.value[20];
		s3.value[9] = number.value[21];
		s3.value[10] = number.value[22];
		s3.value[11] = number.value[23];
		//s4=(c20, c19, c18, c17, c16, c15, c14, c13, c12, c23, c22, c21)
		s4.value[0] = number.value[21];
		s4.value[1] = number.value[22];
		s4.value[2] = number.value[23];
		s4.value[3] = number.value[12];
		s4.value[4] = number.value[13];
		s4.value[5] = number.value[14];
		s4.value[6] = number.value[15];
		s4.value[7] = number.value[16];
		s4.value[8] = number.value[17];
		s4.value[9] = number.value[18];
		s4.value[10] = number.value[19];
		s4.value[11] = number.value[20];
		//s5=(c19, c18, c17, c16, c15, c14, c13, c12, c20,0, c23,0)
		s5.value[0] = 0;
		s5.value[1] = number.value[23];
		s5.value[2] = 0;
		s5.value[3] = number.value[20];
		s5.value[4] = number.value[12];
		s5.value[5] = number.value[13];
		s5.value[6] = number.value[14];
		s5.value[7] = number.value[15];
		s5.value[8] = number.value[16];
		s5.value[9] = number.value[17];
		s5.value[10] = number.value[18];
		s5.value[11] = number.value[19];
		//s6=(0,0,0,0, c23, c22, c21, c20,0,0,0,0)
		s6.value[0] = 0;
		s6.value[1] = 0;
		s6.value[2] = 0;
		s6.value[3] = 0;
		s6.value[4] = number.value[20];
		s6.value[5] = number.value[21];
		s6.value[6] = number.value[22];
		s6.value[7] = number.value[13];
		s6.value[8] = 0;
		s6.value[9] = 0;
		s6.value[10] = 0;
		s6.value[11] = 0;
		//s7=(0,0,0,0,0,0, c23, c22, c21,0,0, c20)
		s7.value[0] = number.value[20];
		s7.value[1] = 0;
		s7.value[2] = 0;
		s7.value[3] = number.value[21];
		s7.value[4] = number.value[22];
		s7.value[5] = number.value[23];
		s7.value[6] = 0;
		s7.value[7] = 0;
		s7.value[8] = 0;
		s7.value[9] = 0;
		s7.value[10] = 0;
		s7.value[11] = 0;
		//s8=(c22, c21, c20, c19, c18, c17, c16, c15, c14, c13, c12, c23)
		s8.value[0] = number.value[23];
		s8.value[1] = number.value[12];
		s8.value[2] = number.value[13];
		s8.value[3] = number.value[14];
		s8.value[4] = number.value[15];
		s8.value[5] = number.value[16];
		s8.value[6] = number.value[17];
		s8.value[7] = number.value[18];
		s8.value[8] = number.value[19];
		s8.value[9] = number.value[20];
		s8.value[10] = number.value[21];
		s8.value[11] = number.value[22];
		//s9=(0,0,0,0,0,0,0, c23, c22, c21, c20,0)
		s9.value[0] = 0;
		s9.value[1] = number.value[20];
		s9.value[2] = number.value[21];
		s9.value[3] = number.value[22];
		s9.value[4] = number.value[23];
		s9.value[5] = 0;
		s9.value[6] = 0;
		s9.value[7] = 0;
		s9.value[8] = 0;
		s9.value[9] = 0;
		s9.value[10] = 0;
		s9.value[11] = 0;
		//s10=(0,0,0,0,0,0,0, c23, c23,0,0,0)
		s10.value[0] = 0;
		s10.value[1] = number.value[20];
		s10.value[2] = number.value[21];
		s10.value[3] = number.value[22];
		s10.value[4] = number.value[23];
		s10.value[5] = 0;
		s10.value[6] = 0;
		s10.value[7] = 0;
		s10.value[8] = 0;
		s10.value[9] = 0;
		s10.value[10] = 0;
		s10.value[11] = 0;

		//result = s1 + 2*s2 + s3 + s4 + s5 + s6 + s7 − s8 − s9 −s10
		tmp1 = pfa::addNum(s2, s2, P384base);				//2*s2. Using add since PTX doesnt have shift with carry.
		tmp2 = pfa::addNum(s1, s3, P384base);				//s1+s3
		tmp3 = pfa::addNum(s4, s5, P384base);				//s4+s5
		tmp4 = pfa::addNum(s6, s7, P384base);				//s6+s7
		tmp5 = pfa::addNum(s8, s9, P384base);				//-s8-s9
		tmp1 = pfa::addNum(tmp1, tmp2, P384base);			//s1+2*s2+s3 
		tmp3 = pfa::addNum(tmp3, tmp4, P384base);			//s4+s5+s6+s7
		tmp5 = pfa::addNum(tmp5, s10, P384base);			//-s8-s9-s10
		tmp1 = pfa::addNum(tmp1, tmp3, P384base);			//s1+2*s2+s3+s4+s5+s6+s7
		result = pfa::substractNum(tmp1, tmp5, P384base);	//s1 + 2*s2 + s3 + s4 + s5 + s6 + s7 − s8 − s9 −s10

		return result;
	}

	//Helper function used in fastRecuctionP521 for topmost 521 bits of input.
	__device__ BigNum<17> shiftRightByNine(BigNum<17> number) {
		BigNum<17> result;
		unsigned int carry1 = 0;
		unsigned int carry2 = 0;

		for (int ind = 16; ind > 0; --ind) {
			carry2 = (number.value[ind] & 0x1ff) << 23;
			result.value[ind] = number.value[ind] >> 9;
			result.value[ind] |= carry1;
			carry1 = carry2;
		}

		result.value[0] = number.value[0] >> 9;
		result.value[0] |= carry1;

		return result;
	}

	__constant__ static const BigNum<17> P521base = { 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
		0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
		0xffffffff, 0x1ff };
	__device__ BigNum<17> fastRecuctionP521(BigNum<34> number) {
		BigNum<17> s1, s2, result;

		//Initialization of s values;
		s1.value[0] = number.value[0];
		s1.value[1] = number.value[1];
		s1.value[2] = number.value[2];
		s1.value[3] = number.value[3];
		s1.value[4] = number.value[4];
		s1.value[5] = number.value[5];
		s1.value[6] = number.value[6];
		s1.value[7] = number.value[7];
		s1.value[8] = number.value[8];
		s1.value[9] = number.value[9];
		s1.value[10] = number.value[10];
		s1.value[11] = number.value[11];
		s1.value[12] = number.value[12];
		s1.value[13] = number.value[13];
		s1.value[14] = number.value[14];
		s1.value[15] = number.value[15];
		s1.value[16] = number.value[16] & 0x1ff;
		s2.value[0] = number.value[16];
		s2.value[1] = number.value[17];
		s2.value[2] = number.value[18];
		s2.value[3] = number.value[19];
		s2.value[4] = number.value[20];
		s2.value[5] = number.value[21];
		s2.value[6] = number.value[22];
		s2.value[7] = number.value[23];
		s2.value[8] = number.value[24];
		s2.value[9] = number.value[25];
		s2.value[10] = number.value[26];
		s2.value[11] = number.value[27];
		s2.value[12] = number.value[28];
		s2.value[13] = number.value[29];
		s2.value[14] = number.value[30];
		s2.value[15] = number.value[31];
		s2.value[16] = number.value[32];

		s2 = shiftRightByNine(s2);

		//result = s1 + s2
		result = pfa::addNum(s1, s2, P521base);
		return result;
	}
	__device__ BigNum<6> multiplyNumAndModP192(BigNum<6> first, BigNum<6> second)
	{
		BigNum<12> tmpRes;

		tmpRes = pfa::multiplyNum(first, second);
		return fastRecuctionP192(tmpRes);
	}

	__device__ BigNum<7> multiplyNumAndModP224(BigNum<7> first, BigNum<7> second)
	{
		BigNum<14> tmpRes;

		tmpRes = pfa::multiplyNum(first, second);
		return fastRecuctionP224(tmpRes);
	}

	__device__ BigNum<8> multiplyNumAndModP256(BigNum<8> first, BigNum<8> second)
	{
		BigNum<16> tmpRes;

		tmpRes = pfa::multiplyNum(first, second);
		return fastRecuctionP256(tmpRes);
	}

	__device__ BigNum<12> multiplyNumAndModP384(BigNum<12> first, BigNum<12> second)
	{
		BigNum<24> tmpRes;

		tmpRes = pfa::multiplyNum(first, second);
		return fastRecuctionP384(tmpRes);
	}

	__device__ BigNum<17> multiplyNumAndModP521(BigNum<17> first, BigNum<17> second)
	{
		BigNum<34> tmpRes;

		tmpRes = pfa::multiplyNum(first, second);
		return fastRecuctionP521(tmpRes);
	}

	__device__ BigNum<6> squareNumAndModP192(BigNum<6> number)
	{
		BigNum<12> tmpRes;

		tmpRes = pfa::squareNum(number);
		return fastRecuctionP192(tmpRes);
	}

	__device__ BigNum<7> squareNumAndModP224(BigNum<7> number)
	{
		BigNum<14> tmpRes;

		tmpRes = pfa::squareNum(number);
		return fastRecuctionP224(tmpRes);
	}

	__device__ BigNum<8> squareNumAndModP256(BigNum<8> number)
	{
		BigNum<16> tmpRes;

		tmpRes = pfa::squareNum(number);
		return fastRecuctionP256(tmpRes);
	}

	__device__ BigNum<12> squareNumAndModP384(BigNum<12> number)
	{
		BigNum<24> tmpRes;

		tmpRes = pfa::squareNum(number);
		return fastRecuctionP384(tmpRes);
	}

	__device__ BigNum<17> squareNumAndModP521(BigNum<17> number)
	{
		BigNum<34> tmpRes;

		tmpRes = pfa::squareNum(number);
		return fastRecuctionP521(tmpRes);
	}

	__device__ BigNum<6> multiplyNumIntAndModP192(BigNum<6> first, unsigned int second)
	{
		BigNum<12> tmpRes;

		tmpRes = pfa::multiplyNumInt(first, second);
		return fastRecuctionP192(tmpRes);
	}
	
	__device__ BigNum<7>  multiplyNumIntAndModP224(BigNum<7> first, unsigned int second)
	{
		BigNum<14> tmpRes;

		tmpRes = pfa::multiplyNumInt(first, second);
		return fastRecuctionP224(tmpRes);
	}

	__device__ BigNum<8> multiplyNumIntAndModP256(BigNum<8> first, unsigned int second)
	{
		BigNum<16> tmpRes;

		tmpRes = pfa::multiplyNumInt(first, second);
		return fastRecuctionP256(tmpRes);
	}

	__device__ BigNum<12> multiplyNumIntAndModP384(BigNum<12> first, unsigned int second)
	{
		BigNum<24> tmpRes;

		tmpRes = pfa::multiplyNumInt(first, second);
		return fastRecuctionP384(tmpRes);
	}

	__device__ BigNum<17> multiplyNumIntAndModP521(BigNum<17> first, unsigned int second)
	{
		BigNum<34> tmpRes;

		tmpRes = pfa::multiplyNumInt(first, second);
		return fastRecuctionP521(tmpRes);
	}

	__constant__ static const WeierstrassCurve<6> secp192r1 = {
		{ 0xffffffff, 0xffffffff, 0xfffffffe, 0xffffffff, 0xffffffff, 0xffffffff },
		{ 0xB4D22831, 0x146BC9B1, 0x99DEF836, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF },
		{ {0x82ff1012, 0xf4ff0afd, 0x43a18800, 0x7cbf20eb, 0xb03090f6, 0x188da80e},
			{ 0x1e794811, 0x73f977a1, 0x6b24cdd5, 0x631011ed, 0xffc8da78, 0x7192b95} },
		{ 0xFFFFFFFC, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF },
		{ 0xC146B9B1, 0xFEB8DEEC, 0x72243049, 0x0FA7E9AB, 0xE59C80E7, 0x64210519 },
		&multiplyNumAndModP192,
		&multiplyNumIntAndModP192,
		&squareNumAndModP192
	};

	__constant__ static const WeierstrassCurve<7> secp224r1 = {
		{ 1, 0, 0, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff },
		{ 0x5C5C2A3D, 0x13DD2945, 0xE0B8F03E, 0xFFFF16A2, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF },
		{ { 0x115C1D21, 0x343280D6, 0x56C21122, 0x4A03C1D3, 0x321390B9, 0x6BB4BF7F, 0xB70E0CBD },
			{ 0x85007E34, 0x44D58199, 0x5A074764, 0xCD4375A0, 0x4C22DFE6, 0xB5F723FB, 0xBD376388 } },
		{ 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF },
		{ 0x2355FFB4, 0x270B3943, 0xD7BFD8BA, 0x5044B0B7, 0xF5413256, 0x0C04B3AB, 0xB4050A85 },
		&multiplyNumAndModP224,
		&multiplyNumIntAndModP224,
		&squareNumAndModP224
	};

	__constant__ static const WeierstrassCurve<8> secp256r1 = {
		{ 0xffffffff, 0xffffffff, 0xffffffff, 0, 0, 0, 0x1, 0xffffffff },
		{ 0xFC632551, 0xF3B9CAC2, 0xA7179E84, 0xBCE6FAAD, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF },
		{ { 0xD898C296, 0xF4A13945, 0x2DEB33A0, 0x77037D81, 0x63A440F2, 0xF8BCE6E5, 0xE12C4247, 0x6B17D1F2 },
			{ 0x37BF51F5, 0xCBB64068, 0x6B315ECE, 0x2BCE3357,  0x7C0F9E16, 0x8EE7EB4A, 0xFE1A7F9B, 0x4FE342E2 } },
		{ 0xFFFFFFFC, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0, 0, 1, 0xFFFFFFFF },
		{ 0x27D2604B, 0x3BCE3C3E, 0xCC53B0F6, 0x651D06B0, 0x769886BC, 0xB3EBBD55, 0xAA3A93E7, 0x5AC635D8 },
		&multiplyNumAndModP256,
		&multiplyNumIntAndModP256,
		&squareNumAndModP256
	};

	__constant__ static const WeierstrassCurve<12> secp384r1 = {
		{ 0xffffffff, 0, 0, 0xffffffff, 0xfffffffe, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff },
		{ 0xCCC52973, 0xECEC196A, 0x48B0A77A, 0x581A0DB2, 0xF4372DDF, 0xC7634D81, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF },
		{ { 0x72760AB7, 0x3A545E38, 0xBF55296C, 0x5502F25D, 0x82542A38, 0x59F741E0, 0x8BA79B98, 0x6E1D3B62, 0xF320AD74, 0x8EB1C71E, 0xBE8B0537, 0xAA87CA22 },
			{ 0x90EA0E5F, 0x7A431D7C, 0x1D7E819D, 0x0A60B1CE, 0xB5F0B8C0, 0xE9DA3113, 0x289A147C, 0xF8F41DBD, 0x9292DC29, 0x5D9E98BF, 0x96262C6F, 0x3617DE4A } },
		{ 0xfffffffc, 0, 0, 0xffffffff, 0xfffffffe, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff },
		{ 0xD3EC2AEF, 0x2A85C8ED, 0xC656398D, 0x5013875A, 0x8A2ED19D, 0x0314088F, 0xFE814112, 0x181D9C6E, 0xE3F82D19, 0x988E056B, 0xE23EE7E4, 0xB3312FA7 },
		&multiplyNumAndModP384,
		&multiplyNumIntAndModP384,
		&squareNumAndModP384
	};

	__constant__ static const WeierstrassCurve<17> secp521r1 = {
		{ 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
		0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x1ff },
		{ 0x91386409, 0xBB6FB71E, 0x899C47AE, 0x3BB5C9B8, 0xF709A5D0, 0x7FCC0148, 0xBF2F966B, 0x51868783, 0xFFFFFFFA, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
		0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x1FF },
		{ { 0xC2E5BD66, 0xF97E7E31, 0x856A429B, 0x3348B3C1, 0xA2FFA8DE, 0xFE1DC127, 0xEFE75928, 0xA14B5E77, 0x6B4D3DBA, 0xF828AF60, 0x053FB521, 0x9C648139,
		0x2395B442, 0x9E3ECB66, 0x0404E9CD, 0x858E06B7, 0xC6 },
			{ 0x9FD16650, 0x88BE9476, 0xA272C240, 0x353C7086, 0x3FAD0761, 0xC550B901, 0x5EF42640, 0x97EE7299, 0x273E662C, 0x17AFBD17, 0x579B4468, 0x98F54449, 
			0x2C7D1BD9, 0x5C8A5FB4, 0x9A3BC004, 0x39296A78, 0x118 } },
		{ 0xfffffffc, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
		0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x1ff },
		{ 0x6B503F00, 0xEF451FD4, 0x3D2C34F1, 0x3573DF88, 0x3BB1BF07, 0x1652C0BD, 0xEC7E937B, 0x56193951, 0x8EF109E1, 0xB8B48991, 0x99B315F3, 0xA2DA725B, 
		0xB68540EE, 0x929A21A0, 0x8E1C9A1F, 0x953EB961, 0x51 },
		&multiplyNumAndModP521,
		&multiplyNumIntAndModP521,
		&squareNumAndModP521
	};
}