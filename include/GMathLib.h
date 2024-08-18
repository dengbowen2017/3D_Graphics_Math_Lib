#pragma once

#include <assert.h>
#include <immintrin.h>
#include <math.h>

namespace GMath
{
	// Constant Definitions
	constexpr float MATH_PI = 3.141592654f;

	constexpr float ConvertToRadians(float Degrees) noexcept { return Degrees * MATH_PI / 180.0f; }
	constexpr float ConvertToDegrees(float Radians) noexcept { return Radians * 180.0f / MATH_PI; }

	// Type Definitions
	struct MVector
	{
		__m128 v;

		MVector operator+() const noexcept { return *this; }
		MVector operator-() const noexcept;
		MVector& __vectorcall operator+=(const MVector V) noexcept;
		MVector& __vectorcall operator-=(const MVector V) noexcept;
		MVector& __vectorcall operator*=(const MVector V) noexcept;
		MVector& __vectorcall operator/=(const MVector V) noexcept;
		MVector& operator*=(float S) noexcept;
		MVector& operator/=(float S) noexcept;
		MVector __vectorcall operator+(const MVector V) const noexcept;
		MVector __vectorcall operator-(const MVector V) const noexcept;
		MVector __vectorcall operator*(const MVector V) const noexcept;
		MVector __vectorcall operator/(const MVector V) const noexcept;
		MVector operator*(float S) const noexcept;
		MVector operator/(float S) const noexcept;
		friend MVector __vectorcall operator*(float S, const MVector V) noexcept;
	};

	struct MMatrix
	{
		__m128 r[4];

		MMatrix operator+() const noexcept { return *this; }
		MMatrix operator-() const noexcept;
		MMatrix& __vectorcall operator+=(const MMatrix M) noexcept;
		MMatrix& __vectorcall operator-=(const MMatrix M) noexcept;
		MMatrix& __vectorcall operator*=(const MMatrix M) noexcept;
		MMatrix& operator*=(float S) noexcept;
		MMatrix& operator/=(float S) noexcept;
		MMatrix __vectorcall operator+(const MMatrix M) const noexcept;
		MMatrix __vectorcall operator-(const MMatrix M) const noexcept;
		MMatrix __vectorcall operator*(const MMatrix M) const noexcept;
		MMatrix operator*(float S) const noexcept;
		MMatrix operator/(float S) const noexcept;
		friend MMatrix __vectorcall operator*(float S, const MMatrix M) noexcept;
	};

	struct Vector3
	{
		float x;
		float y;
		float z;

		Vector3() = default;
		constexpr Vector3(float init) noexcept : x(init), y(init), z(init) {};
		constexpr Vector3(float _x, float _y, float _z) noexcept : x(_x), y(_y), z(_z) {};
	};

	__declspec(align(16)) struct Vector3A : public Vector3
	{
		using Vector3::Vector3;
	};

	struct Vector4
	{
		float x;
		float y;
		float z;
		float w;

		Vector4() = default;
		constexpr Vector4(float init) noexcept : x(init), y(init), z(init), w(init) {};
		constexpr Vector4(float _x, float _y, float _z, float _w) noexcept : x(_x), y(_y), z(_z), w(_w) {};
	};

	__declspec(align(16)) struct Vector4A : public Vector4
	{
		using Vector4::Vector4;
	};

	struct Matrix3x3
	{
		float m[3][3];

		float operator() (unsigned int column, unsigned int row) const noexcept
		{
			assert(column < 3 && row < 3);
			return m[column][row];
		}

		float& operator() (unsigned int column, unsigned int row) noexcept
		{
			assert(column < 3 && row < 3);
			return m[column][row];
		}
	};

	struct Matrix4x4
	{
		float m[4][4];

		float operator() (unsigned int column, unsigned int row) const noexcept
		{
			assert(column < 4 && row < 4);
			return m[column][row];
		}

		float& operator() (unsigned int column, unsigned int row) noexcept
		{
			assert(column < 4 && row < 4);
			return m[column][row];
		}
	};

	__declspec(align(16)) struct Matrix4x4A : public Matrix4x4
	{

	};

	// Global Values
	extern const __declspec(selectany) __m128 g_MatIdentityR0 = _mm_set_ps(0, 0, 0, 1.0f);
	extern const __declspec(selectany) __m128 g_MatIdentityR1 = _mm_set_ps(0, 0, 1.0f, 0);
	extern const __declspec(selectany) __m128 g_MatIdentityR2 = _mm_set_ps(0, 1.0f, 0, 0);
	extern const __declspec(selectany) __m128 g_MatIdentityR3 = _mm_set_ps(1.0f, 0, 0, 0);

																						//w			//z			//y			//x						
	extern const __declspec(selectany) __m128 g_VecMaskX      = *(__m128*)&_mm_set_epi32(0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF);
	extern const __declspec(selectany) __m128 g_VecMaskY      = *(__m128*)&_mm_set_epi32(0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000);
	extern const __declspec(selectany) __m128 g_VecMaskZ      = *(__m128*)&_mm_set_epi32(0x00000000, 0xFFFFFFFF, 0x00000000, 0x00000000);
	extern const __declspec(selectany) __m128 g_VecMaskW      = *(__m128*)&_mm_set_epi32(0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000);
	extern const __declspec(selectany) __m128 g_VecMaskZW     = *(__m128*)&_mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000);
	extern const __declspec(selectany) __m128 g_VecMaskXYZ    = *(__m128*)&_mm_set_epi32(0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);

	// Miscellaneous Operations
	float ScalarSin(float Radians) noexcept { return sin(Radians); }
	float ScalarCos(float Radians) noexcept { return cos(Radians); }

	// Load Operations
	MVector __vectorcall LoadVector3(const Vector3* p_src) noexcept
	{
		assert(p_src);
		MVector V;
		__m128 xy = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(p_src)));
		__m128 z = _mm_load_ss(&p_src->z);
		V.v = _mm_movelh_ps(xy, z);
		return V;
	}

	MVector __vectorcall LoadVector3A(const Vector3A* p_src) noexcept
	{
		assert(p_src);
		assert((reinterpret_cast<uintptr_t>(p_src) & 0xF) == 0);
		MVector V;
		__m128 xyz = _mm_load_ps(&p_src->x);
		V.v = _mm_and_ps(xyz, g_VecMaskXYZ);
		return V;
	}

	MVector __vectorcall LoadVector4(const Vector4* p_src) noexcept
	{
		assert(p_src);
		MVector V;
		V.v = _mm_loadu_ps(&p_src->x);
		return V;
	}

	MVector __vectorcall LoadVector4A(const Vector4A* p_src) noexcept
	{
		assert(p_src);
		assert((reinterpret_cast<uintptr_t>(p_src) & 0xF) == 0);
		MVector V;
		V.v = _mm_load_ps(&p_src->x);
		return V;
	}

	MMatrix __vectorcall LoadMatrix3x3(const Matrix3x3* p_src) noexcept
	{
		assert(p_src);
		__m128 Z = _mm_setzero_ps();

		__m128 V1 = _mm_loadu_ps(&p_src->m[0][0]);
		__m128 V2 = _mm_loadu_ps(&p_src->m[1][1]);
		__m128 V3 = _mm_load_ss(&p_src->m[2][2]);

		__m128 T1 = _mm_unpackhi_ps(V1, Z);
		__m128 T2 = _mm_unpacklo_ps(V2, Z);
		__m128 T3 = _mm_shuffle_ps(V3, T2, _MM_SHUFFLE(0, 1, 0, 0));
		__m128 T4 = _mm_movehl_ps(T2, T3);
		__m128 T5 = _mm_movehl_ps(Z, T1);

		MMatrix M;
		M.r[0] = _mm_movelh_ps(V1, T1);
		M.r[1] = _mm_add_ps(T4, T5);
		M.r[2] = _mm_shuffle_ps(V2, V3, _MM_SHUFFLE(1, 0, 3, 2));
		M.r[3] = g_MatIdentityR3;
		return M;
	}

	MMatrix __vectorcall LoadMatrix4x4(const Matrix4x4* p_src) noexcept
	{
		assert(p_src);
		MMatrix M;
		M.r[0] = _mm_loadu_ps(&p_src->m[0][0]);
		M.r[1] = _mm_loadu_ps(&p_src->m[1][0]);
		M.r[2] = _mm_loadu_ps(&p_src->m[2][0]);
		M.r[3] = _mm_loadu_ps(&p_src->m[3][0]);
		return M;
	}

	MMatrix __vectorcall LoadMatrix4x4A(const Matrix4x4A* p_src) noexcept
	{
		assert(p_src);
		assert((reinterpret_cast<uintptr_t>(p_src) & 0xF) == 0);
		MMatrix M;
		M.r[0] = _mm_load_ps(&p_src->m[0][0]);
		M.r[1] = _mm_load_ps(&p_src->m[1][0]);
		M.r[2] = _mm_load_ps(&p_src->m[2][0]);
		M.r[3] = _mm_load_ps(&p_src->m[3][0]);
		return M;
	}

	// Store Operations
	void __vectorcall StoreVector3(Vector3* p_des, const MVector V) noexcept
	{
		assert(p_des);
		_mm_store_sd(reinterpret_cast<double*>(p_des), _mm_castps_pd(V.v));
		__m128 z = _mm_shuffle_ps(V.v, V.v, _MM_SHUFFLE(2, 2, 2, 2));
		_mm_store_ss(&p_des->z, z);
	}

	void __vectorcall StoreVector3A(Vector3A* p_des, const MVector V) noexcept
	{
		assert(p_des);
		assert((reinterpret_cast<uintptr_t>(p_des) & 0xF) == 0);
		_mm_store_sd(reinterpret_cast<double*>(p_des), _mm_castps_pd(V.v));
		__m128 z = _mm_movehl_ps(V.v, V.v);
		_mm_store_ss(&p_des->z, z);
	}

	void __vectorcall StoreVector4(Vector4* p_des, const MVector V) noexcept
	{
		assert(p_des);
		_mm_storeu_ps(&p_des->x, V.v);
	}

	void __vectorcall StoreVector4A(Vector4A* p_des, const MVector V) noexcept
	{
		assert(p_des);
		assert((reinterpret_cast<uintptr_t>(p_des) & 0xF) == 0);
		_mm_store_ps(&p_des->x, V.v);
	}

	void __vectorcall StoreMatrix3x3(Matrix3x3* p_des, const MMatrix M) noexcept
	{
		assert(p_des);
		__m128 vTemp1 = M.r[0];
		__m128 vTemp2 = M.r[1];
		__m128 vTemp3 = M.r[2];
		__m128 vWork = _mm_shuffle_ps(vTemp1, vTemp2, _MM_SHUFFLE(0, 0, 2, 2));
		vTemp1 = _mm_shuffle_ps(vTemp1, vWork, _MM_SHUFFLE(2, 0, 1, 0));
		_mm_storeu_ps(&p_des->m[0][0], vTemp1);
		vTemp2 = _mm_shuffle_ps(vTemp2, vTemp3, _MM_SHUFFLE(1, 0, 2, 1));
		_mm_storeu_ps(&p_des->m[1][1], vTemp2);
		vTemp3 = _mm_shuffle_ps(vTemp3, vTemp3, _MM_SHUFFLE(2, 2, 2, 2));
		_mm_store_ss(&p_des->m[2][2], vTemp3);
	}

	void __vectorcall StoreMatrix4x4(Matrix4x4* p_des, const MMatrix M) noexcept
	{
		assert(p_des);
		_mm_storeu_ps(&p_des->m[0][0], M.r[0]);
		_mm_storeu_ps(&p_des->m[1][0], M.r[1]);
		_mm_storeu_ps(&p_des->m[2][0], M.r[2]);
		_mm_storeu_ps(&p_des->m[3][0], M.r[3]);
	}

	void __vectorcall StoreMatrix4x4A(Matrix4x4A* p_des, const MMatrix M) noexcept
	{
		assert(p_des);
		assert((reinterpret_cast<uintptr_t>(p_des) & 0xF) == 0);
		_mm_store_ps(&p_des->m[0][0], M.r[0]);
		_mm_store_ps(&p_des->m[1][0], M.r[1]);
		_mm_store_ps(&p_des->m[2][0], M.r[2]);
		_mm_store_ps(&p_des->m[3][0], M.r[3]);
	}

	// Vector Operations
	MVector __vectorcall VectorNegate(const MVector V) noexcept
	{
		MVector vRes;
		__m128 Z = _mm_setzero_ps();
		vRes.v = _mm_sub_ps(Z, V.v);
		return vRes;
	}

	MVector __vectorcall VectorAdd(const MVector V1, const MVector V2) noexcept
	{
		MVector vRes;
		vRes.v = _mm_add_ps(V1.v, V2.v);
		return vRes;
	}

	MVector __vectorcall VectorSub(const MVector V1, const MVector V2) noexcept
	{
		MVector vRes;
		vRes.v = _mm_sub_ps(V1.v, V2.v);
		return vRes;
	}

	MVector __vectorcall VectorMul(const MVector V1, const MVector V2) noexcept
	{
		MVector vRes;
		vRes.v = _mm_mul_ps(V1.v, V2.v);
		return vRes;
	}

	MVector __vectorcall VectorDiv(const MVector V1, const MVector V2) noexcept
	{
		MVector vRes;
		vRes.v = _mm_div_ps(V1.v, V2.v);
		return vRes;
	}

	MVector __vectorcall VectorSum(const MVector V) noexcept
	{
		MVector vRes;
		__m128 vTemp = _mm_hadd_ps(V.v, V.v);
		vRes.v = _mm_hadd_ps(vTemp, vTemp);
		return vRes;
	}

	MVector __vectorcall VectorScale(const MVector V, float S) noexcept
	{
		MVector vRes;
		__m128 vS = _mm_set_ps1(S);
		vRes.v = _mm_mul_ps(vS, V.v);
		return vRes;
	}

	MVector __vectorcall VectorNormalize(const MVector V) noexcept
	{
		MVector vRes;
		__m128 vLengthSq = _mm_dp_ps(V.v, V.v, 0xff);
		__m128 vResult = _mm_sqrt_ps(vLengthSq);
		__m128 vZeroMask = _mm_setzero_ps();
		vZeroMask = _mm_cmpneq_ps(vZeroMask, vResult);
		vResult = _mm_div_ps(V.v, vResult);
		vRes.v = _mm_and_ps(vResult, vZeroMask);
		return vRes;
	}

	MVector __vectorcall VectorDot(const MVector V1, const MVector V2) noexcept
	{
		MVector vRes;
		vRes.v = _mm_dp_ps(V1.v, V2.v, 0xff);
		return vRes;
	}

	MVector __vectorcall VectorCross(const MVector V1, const MVector V2) noexcept
	{
		MVector vRes;
		__m128 vTemp1 = _mm_shuffle_ps(V1.v, V1.v, _MM_SHUFFLE(3, 0, 2, 1));
		__m128 vTemp2 = _mm_shuffle_ps(V2.v, V2.v, _MM_SHUFFLE(3, 1, 0, 2));
		__m128 vResult = _mm_mul_ps(vTemp1, vTemp2);
		vTemp1 = _mm_shuffle_ps(V1.v, V1.v, _MM_SHUFFLE(3, 1, 0, 2));
		vTemp2 = _mm_shuffle_ps(V2.v, V2.v, _MM_SHUFFLE(3, 0, 2, 1));
		vResult = _mm_sub_ps(vResult, _mm_mul_ps(vTemp1, vTemp2));
		vRes.v = _mm_and_ps(vResult, g_VecMaskXYZ);
		return vRes;
	}

	// Vector Overloads
	MVector MVector::operator-() const noexcept
	{
		return VectorNegate(*this);
	}

	MVector& __vectorcall MVector::operator+=(const MVector V) noexcept
	{
		this->v = _mm_add_ps(this->v, V.v);
		return *this;
	}

	MVector& __vectorcall MVector::operator-=(const MVector V) noexcept
	{
		this->v = _mm_sub_ps(this->v, V.v);
		return *this;
	}

	MVector& __vectorcall MVector::operator*=(const MVector V) noexcept
	{
		this->v = _mm_mul_ps(this->v, V.v);
		return *this;
	}

	MVector& __vectorcall MVector::operator/=(const MVector V) noexcept
	{
		this->v = _mm_div_ps(this->v, V.v);
		return *this;
	}

	MVector& MVector::operator*=(float S) noexcept
	{
		__m128 vS = _mm_set_ps1(S);
		this->v = _mm_mul_ps(this->v, vS);
		return *this;
	}

	MVector& MVector::operator/=(float S) noexcept
	{
		__m128 vS = _mm_set_ps1(S);
		this->v = _mm_div_ps(this->v, vS);
		return *this;
	}

	MVector __vectorcall MVector::operator+(const MVector V) const noexcept
	{
		return VectorAdd(*this, V);
	}

	MVector __vectorcall MVector::operator-(const MVector V) const noexcept
	{
		return VectorSub(*this, V);
	}

	MVector __vectorcall MVector::operator*(const MVector V) const noexcept
	{
		return VectorMul(*this, V);
	}

	MVector __vectorcall MVector::operator/(const MVector V) const noexcept
	{
		return VectorDiv(*this, V);
	}

	MVector MVector::operator*(float S) const noexcept
	{
		return VectorScale(*this, S);
	}

	MVector MVector::operator/(float S) const noexcept
	{
		MVector vRes;
		__m128 vS = _mm_set_ps1(S);
		vRes.v = _mm_div_ps(this->v, vS);
		return vRes;
	}

	MVector __vectorcall operator*(float S, const MVector V) noexcept
	{
		return VectorScale(V, S);
	}

	// Matrix Operation
	MMatrix __vectorcall MatrixMul(const MMatrix M1, const MMatrix& M2) noexcept
	{
		MMatrix mRes;

		__m128 vX = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[0]) + 0);
		__m128 vY = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[0]) + 1);
		__m128 vZ = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[0]) + 2);
		__m128 vW = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[0]) + 3);

		vX = _mm_mul_ps(vX, M2.r[0]);
		vY = _mm_mul_ps(vY, M2.r[1]);
		vZ = _mm_mul_ps(vZ, M2.r[2]);
		vW = _mm_mul_ps(vW, M2.r[3]);

		vX = _mm_add_ps(vX, vZ);
		vY = _mm_add_ps(vY, vW);
		vX = _mm_add_ps(vX, vY);
		mRes.r[0] = vX;

		vX = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[1]) + 0);
		vY = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[1]) + 1);
		vZ = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[1]) + 2);
		vW = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[1]) + 3);

		vX = _mm_mul_ps(vX, M2.r[0]);
		vY = _mm_mul_ps(vY, M2.r[1]);
		vZ = _mm_mul_ps(vZ, M2.r[2]);
		vW = _mm_mul_ps(vW, M2.r[3]);

		vX = _mm_add_ps(vX, vZ);
		vY = _mm_add_ps(vY, vW);
		vX = _mm_add_ps(vX, vY);
		mRes.r[1] = vX;

		vX = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[2]) + 0);
		vY = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[2]) + 1);
		vZ = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[2]) + 2);
		vW = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[2]) + 3);

		vX = _mm_mul_ps(vX, M2.r[0]);
		vY = _mm_mul_ps(vY, M2.r[1]);
		vZ = _mm_mul_ps(vZ, M2.r[2]);
		vW = _mm_mul_ps(vW, M2.r[3]);

		vX = _mm_add_ps(vX, vZ);
		vY = _mm_add_ps(vY, vW);
		vX = _mm_add_ps(vX, vY);
		mRes.r[2] = vX;

		vX = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[3]) + 0);
		vY = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[3]) + 1);
		vZ = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[3]) + 2);
		vW = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[3]) + 3);

		vX = _mm_mul_ps(vX, M2.r[0]);
		vY = _mm_mul_ps(vY, M2.r[1]);
		vZ = _mm_mul_ps(vZ, M2.r[2]);
		vW = _mm_mul_ps(vW, M2.r[3]);

		vX = _mm_add_ps(vX, vZ);
		vY = _mm_add_ps(vY, vW);
		vX = _mm_add_ps(vX, vY);
		mRes.r[3] = vX;
		return mRes;
	}

	MMatrix __vectorcall MatrixLookAtRH(const MVector EyePos, const MVector FocusPos, const MVector UpDir) noexcept
	{
		MVector R2 = VectorNormalize(FocusPos - EyePos);
		MVector R0 = VectorNormalize(VectorCross(R2, UpDir));
		MVector R1 = VectorNormalize(VectorCross(R0, R2));
		MVector NegEyePos = VectorNegate(EyePos);

		MVector D0 = VectorDot(R0, NegEyePos);
		MVector D1 = VectorDot(R1, NegEyePos);
		MVector D2 = VectorDot(R2, NegEyePos);

		__m128 vZ = _mm_setzero_ps();

		MMatrix M;
		M.r[0] = _mm_blend_ps(D0.v, R0.v, 0x07);
		M.r[1] = _mm_blend_ps(D1.v, R1.v, 0x07);
		M.r[2] = _mm_sub_ps(vZ, _mm_blend_ps(D2.v, R2.v, 0x07));
		M.r[3] = g_MatIdentityR3;

		return M;
	}

	MMatrix __vectorcall MatrixPerspectiveFovRH(float FovY, float AspectRation, float NearZ, float FarZ) noexcept
	{
		float fRange = FarZ / (NearZ - FarZ);
		float Height = ScalarCos(FovY) / ScalarSin(FovY);
		__m128 V = _mm_set_ps(fRange * NearZ, fRange, Height, Height / AspectRation);

		MMatrix M;
		M.r[0] = _mm_and_ps(V, g_VecMaskX);
		M.r[1] = _mm_and_ps(V, g_VecMaskY);
		M.r[2] = _mm_and_ps(V, g_VecMaskZW);
		M.r[3] = _mm_set_ps(0.0f, -1.0f, 0.0f, 0.0f);

		return M;
	}

	MMatrix __vectorcall MatrixOrthographicRH(float ViewWidth, float ViewHeight, float NearZ, float FarZ) noexcept
	{
		float fRange = 1.0f / (NearZ - FarZ);
		__m128 V = _mm_set_ps(fRange * NearZ, fRange, 2.0f / ViewHeight, 2.0f / ViewWidth);

		MMatrix M;
		M.r[0] = _mm_and_ps(V, g_VecMaskX);
		M.r[1] = _mm_and_ps(V, g_VecMaskY);
		M.r[2] = _mm_and_ps(V, g_VecMaskZW);
		M.r[3] = g_MatIdentityR3;

		return M;
	}

	// Matrix Overloads
	MMatrix MMatrix::operator-() const noexcept
	{
		MMatrix M;
		__m128 Z = _mm_setzero_ps();
		M.r[0] = _mm_sub_ps(Z, this->r[0]);
		M.r[1] = _mm_sub_ps(Z, this->r[1]);
		M.r[2] = _mm_sub_ps(Z, this->r[2]);
		M.r[3] = _mm_sub_ps(Z, this->r[3]);
		return M;
	}

	MMatrix& __vectorcall MMatrix::operator+=(const MMatrix M) noexcept
	{
		this->r[0] = _mm_add_ps(this->r[0], M.r[0]);
		this->r[1] = _mm_add_ps(this->r[0], M.r[1]);
		this->r[2] = _mm_add_ps(this->r[0], M.r[2]);
		this->r[3] = _mm_add_ps(this->r[0], M.r[3]);
		return *this;
	}

	MMatrix& __vectorcall MMatrix::operator-=(const MMatrix M) noexcept
	{
		this->r[0] = _mm_sub_ps(this->r[0], M.r[0]);
		this->r[1] = _mm_sub_ps(this->r[0], M.r[1]);
		this->r[2] = _mm_sub_ps(this->r[0], M.r[2]);
		this->r[3] = _mm_sub_ps(this->r[0], M.r[3]);
		return *this;
	}

	MMatrix& __vectorcall MMatrix::operator*=(const MMatrix M) noexcept
	{
		*this = MatrixMul(*this, M);
		return *this;
	}

	MMatrix& MMatrix::operator*=(float S) noexcept
	{
		__m128 vS = _mm_set_ps1(S);
		this->r[0] = _mm_mul_ps(this->r[0], vS);
		this->r[1] = _mm_mul_ps(this->r[1], vS);
		this->r[2] = _mm_mul_ps(this->r[2], vS);
		this->r[3] = _mm_mul_ps(this->r[3], vS);
		return *this;
	}

	MMatrix& MMatrix::operator/=(float S) noexcept
	{
		__m128 vS = _mm_set_ps1(S);
		this->r[0] = _mm_div_ps(this->r[0], vS);
		this->r[1] = _mm_div_ps(this->r[1], vS);
		this->r[2] = _mm_div_ps(this->r[2], vS);
		this->r[3] = _mm_div_ps(this->r[3], vS);
		return *this;
	}

	MMatrix __vectorcall MMatrix::operator+(const MMatrix M) const noexcept
	{
		MMatrix mRes;
		mRes.r[0] = _mm_add_ps(this->r[0], M.r[0]);
		mRes.r[1] = _mm_add_ps(this->r[1], M.r[1]);
		mRes.r[2] = _mm_add_ps(this->r[2], M.r[2]);
		mRes.r[3] = _mm_add_ps(this->r[3], M.r[3]);
		return mRes;
	}

	MMatrix __vectorcall MMatrix::operator-(const MMatrix M) const noexcept
	{
		MMatrix mRes;
		mRes.r[0] = _mm_sub_ps(this->r[0], M.r[0]);
		mRes.r[1] = _mm_sub_ps(this->r[1], M.r[1]);
		mRes.r[2] = _mm_sub_ps(this->r[2], M.r[2]);
		mRes.r[3] = _mm_sub_ps(this->r[3], M.r[3]);
		return mRes;
	}

	MMatrix __vectorcall MMatrix::operator*(const MMatrix M) const noexcept
	{
		return MatrixMul(*this, M);
	}

	MMatrix MMatrix::operator*(float S) const noexcept
	{
		MMatrix mRes;
		__m128 vS = _mm_set_ps1(S);
		mRes.r[0] = _mm_mul_ps(this->r[0], vS);
		mRes.r[1] = _mm_mul_ps(this->r[1], vS);
		mRes.r[2] = _mm_mul_ps(this->r[2], vS);
		mRes.r[3] = _mm_mul_ps(this->r[3], vS);
		return mRes;
	}

	MMatrix MMatrix::operator/(float S) const noexcept
	{
		MMatrix mRes;
		__m128 vS = _mm_set_ps1(S);
		mRes.r[0] = _mm_div_ps(this->r[0], vS);
		mRes.r[1] = _mm_div_ps(this->r[1], vS);
		mRes.r[2] = _mm_div_ps(this->r[2], vS);
		mRes.r[3] = _mm_div_ps(this->r[3], vS);
		return mRes;
	}

	MMatrix __vectorcall operator*(float S, const MMatrix M) noexcept
	{
		MMatrix mRes;
		__m128 vS = _mm_set_ps1(S);
		mRes.r[0] = _mm_mul_ps(M.r[0], vS);
		mRes.r[1] = _mm_mul_ps(M.r[1], vS);
		mRes.r[2] = _mm_mul_ps(M.r[2], vS);
		mRes.r[3] = _mm_mul_ps(M.r[3], vS);
		return mRes;
	}

#include "GMathVector.inl"
#include "GMathMatrix.inl"
#include "GMathQuaternion.inl"

}