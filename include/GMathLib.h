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

		MVector() = default;
		MVector(float x) noexcept : v(_mm_set_ps1(x)) {}
		MVector(float x, float y, float z, float w) noexcept : v(_mm_set_ps(w, z, y, x)) {}

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

	struct MQuaternion
	{
		__m128 q; // (Im, Re)

		MQuaternion() = default;
		MQuaternion(float x, float y, float z, float w) noexcept : q(_mm_set_ps(w, z, y, x)) {}

		MQuaternion operator+() const noexcept { return *this; }
		MQuaternion operator-() const noexcept;
		MQuaternion& __vectorcall operator+=(const MQuaternion Q) noexcept;
		MQuaternion& __vectorcall operator-=(const MQuaternion Q) noexcept;
		MQuaternion& __vectorcall operator*=(const MQuaternion Q) noexcept;
		MQuaternion& operator*=(float S) noexcept;
		MQuaternion& operator/=(float S) noexcept;
		MQuaternion __vectorcall operator+(const MQuaternion Q) const noexcept;
		MQuaternion __vectorcall operator-(const MQuaternion Q) const noexcept;
		MQuaternion __vectorcall operator*(const MQuaternion Q) const noexcept;
		MQuaternion operator*(float S) const noexcept;
		MQuaternion operator/(float S) const noexcept;
		friend MQuaternion __vectorcall operator*(float S, const MQuaternion Q) noexcept;
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

		MVector __vectorcall operator*(const MVector V) const noexcept;
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
						
	extern const __declspec(selectany) __m128 g_VecMaskX	= *(__m128*) & _mm_set_epi32(0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF); // wzyx
	extern const __declspec(selectany) __m128 g_VecMaskY	= *(__m128*) & _mm_set_epi32(0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000);
	extern const __declspec(selectany) __m128 g_VecMaskZ	= *(__m128*) & _mm_set_epi32(0x00000000, 0xFFFFFFFF, 0x00000000, 0x00000000);
	extern const __declspec(selectany) __m128 g_VecMaskW	= *(__m128*) & _mm_set_epi32(0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000);
	extern const __declspec(selectany) __m128 g_VecMaskZW	= *(__m128*) & _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000);
	extern const __declspec(selectany) __m128 g_VecMaskXYZ	= *(__m128*) & _mm_set_epi32(0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);

	// Miscellaneous Operations
	inline float ScalarSin(float Radians) noexcept { return sin(Radians); }
	inline float ScalarCos(float Radians) noexcept { return cos(Radians); }

	// Load Operations
	inline MVector __vectorcall LoadVector3(const Vector3* p_src) noexcept
	{
		assert(p_src);
		MVector V;
		__m128 xy = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(p_src)));
		__m128 z = _mm_load_ss(&p_src->z);
		V.v = _mm_movelh_ps(xy, z);
		return V;
	}

	inline MVector __vectorcall LoadVector3A(const Vector3A* p_src) noexcept
	{
		assert(p_src);
		assert((reinterpret_cast<uintptr_t>(p_src) & 0xF) == 0);
		MVector V;
		__m128 xyz = _mm_load_ps(&p_src->x);
		V.v = _mm_and_ps(xyz, g_VecMaskXYZ);
		return V;
	}

	inline MVector __vectorcall LoadVector4(const Vector4* p_src) noexcept
	{
		assert(p_src);
		MVector V;
		V.v = _mm_loadu_ps(&p_src->x);
		return V;
	}

	inline MVector __vectorcall LoadVector4A(const Vector4A* p_src) noexcept
	{
		assert(p_src);
		assert((reinterpret_cast<uintptr_t>(p_src) & 0xF) == 0);
		MVector V;
		V.v = _mm_load_ps(&p_src->x);
		return V;
	}

	inline MMatrix __vectorcall LoadMatrix3x3(const Matrix3x3* p_src) noexcept
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

	inline MMatrix __vectorcall LoadMatrix4x4(const Matrix4x4* p_src) noexcept
	{
		assert(p_src);
		MMatrix M;
		M.r[0] = _mm_loadu_ps(&p_src->m[0][0]);
		M.r[1] = _mm_loadu_ps(&p_src->m[1][0]);
		M.r[2] = _mm_loadu_ps(&p_src->m[2][0]);
		M.r[3] = _mm_loadu_ps(&p_src->m[3][0]);
		return M;
	}

	inline MMatrix __vectorcall LoadMatrix4x4A(const Matrix4x4A* p_src) noexcept
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
	inline void __vectorcall StoreVector3(Vector3* p_des, const MVector V) noexcept
	{
		assert(p_des);
		_mm_store_sd(reinterpret_cast<double*>(p_des), _mm_castps_pd(V.v));
		__m128 z = _mm_shuffle_ps(V.v, V.v, _MM_SHUFFLE(2, 2, 2, 2));
		_mm_store_ss(&p_des->z, z);
	}

	inline void __vectorcall StoreVector3A(Vector3A* p_des, const MVector V) noexcept
	{
		assert(p_des);
		assert((reinterpret_cast<uintptr_t>(p_des) & 0xF) == 0);
		_mm_store_sd(reinterpret_cast<double*>(p_des), _mm_castps_pd(V.v));
		__m128 z = _mm_movehl_ps(V.v, V.v);
		_mm_store_ss(&p_des->z, z);
	}

	inline void __vectorcall StoreVector4(Vector4* p_des, const MVector V) noexcept
	{
		assert(p_des);
		_mm_storeu_ps(&p_des->x, V.v);
	}

	inline void __vectorcall StoreVector4A(Vector4A* p_des, const MVector V) noexcept
	{
		assert(p_des);
		assert((reinterpret_cast<uintptr_t>(p_des) & 0xF) == 0);
		_mm_store_ps(&p_des->x, V.v);
	}

	inline void __vectorcall StoreMatrix3x3(Matrix3x3* p_des, const MMatrix M) noexcept
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

	inline void __vectorcall StoreMatrix4x4(Matrix4x4* p_des, const MMatrix M) noexcept
	{
		assert(p_des);
		_mm_storeu_ps(&p_des->m[0][0], M.r[0]);
		_mm_storeu_ps(&p_des->m[1][0], M.r[1]);
		_mm_storeu_ps(&p_des->m[2][0], M.r[2]);
		_mm_storeu_ps(&p_des->m[3][0], M.r[3]);
	}

	inline void __vectorcall StoreMatrix4x4A(Matrix4x4A* p_des, const MMatrix M) noexcept
	{
		assert(p_des);
		assert((reinterpret_cast<uintptr_t>(p_des) & 0xF) == 0);
		_mm_store_ps(&p_des->m[0][0], M.r[0]);
		_mm_store_ps(&p_des->m[1][0], M.r[1]);
		_mm_store_ps(&p_des->m[2][0], M.r[2]);
		_mm_store_ps(&p_des->m[3][0], M.r[3]);
	}

	// Vector Operations
	inline float __vectorcall VectorGetX(const MVector V) noexcept
	{
		return _mm_cvtss_f32(V.v);
	}

	inline MVector __vectorcall VectorNegate(const MVector V) noexcept
	{
		MVector vRes;
		__m128 Z = _mm_setzero_ps();
		vRes.v = _mm_sub_ps(Z, V.v);
		return vRes;
	}

	inline MVector __vectorcall VectorAdd(const MVector V1, const MVector V2) noexcept
	{
		MVector vRes;
		vRes.v = _mm_add_ps(V1.v, V2.v);
		return vRes;
	}

	inline MVector __vectorcall VectorSub(const MVector V1, const MVector V2) noexcept
	{
		MVector vRes;
		vRes.v = _mm_sub_ps(V1.v, V2.v);
		return vRes;
	}

	inline MVector __vectorcall VectorMul(const MVector V1, const MVector V2) noexcept
	{
		MVector vRes;
		vRes.v = _mm_mul_ps(V1.v, V2.v);
		return vRes;
	}

	inline MVector __vectorcall VectorDiv(const MVector V1, const MVector V2) noexcept
	{
		MVector vRes;
		vRes.v = _mm_div_ps(V1.v, V2.v);
		return vRes;
	}

	inline MVector __vectorcall VectorSum(const MVector V) noexcept
	{
		MVector vRes;
		__m128 vTemp = _mm_hadd_ps(V.v, V.v);
		vRes.v = _mm_hadd_ps(vTemp, vTemp);
		return vRes;
	}

	inline MVector __vectorcall VectorScale(const MVector V, float S) noexcept
	{
		MVector vRes;
		__m128 vS = _mm_set_ps1(S);
		vRes.v = _mm_mul_ps(vS, V.v);
		return vRes;
	}

	inline MVector __vectorcall VectorMagnitude(const MVector V) noexcept
	{
		__m128 vTemp = _mm_dp_ps(V.v, V.v, 0xff);
		MVector vRes;
		vRes.v = _mm_sqrt_ps(vTemp);
		return vRes;
	}

	inline MVector __vectorcall VectorNormalize(const MVector V) noexcept
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

	inline MVector __vectorcall VectorDot(const MVector V1, const MVector V2) noexcept
	{
		MVector vRes;
		vRes.v = _mm_dp_ps(V1.v, V2.v, 0xff);
		return vRes;
	}

	inline MVector __vectorcall VectorCross(const MVector V1, const MVector V2) noexcept
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

	inline MVector __vectorcall MatrixMulVec(const MMatrix M, const MVector V) noexcept
	{
		MVector vRes;

		__m128 vX = _mm_shuffle_ps(V.v, V.v, _MM_SHUFFLE(0, 0, 0, 0));
		__m128 vY = _mm_shuffle_ps(V.v, V.v, _MM_SHUFFLE(1, 1, 1, 1));
		__m128 vZ = _mm_shuffle_ps(V.v, V.v, _MM_SHUFFLE(2, 2, 2, 2));
		__m128 vW = _mm_shuffle_ps(V.v, V.v, _MM_SHUFFLE(3, 3, 3, 3));

		vX = _mm_mul_ps(vX, M.r[0]);
		vY = _mm_mul_ps(vY, M.r[1]);
		vZ = _mm_mul_ps(vZ, M.r[2]);
		vW = _mm_mul_ps(vW, M.r[3]);

		vX = _mm_add_ps(vX, vZ);
		vY = _mm_add_ps(vY, vW);
		vX = _mm_add_ps(vX, vY);

		vRes.v = vX;
		return vRes;
	}

	// Vector Overloads
	inline MVector MVector::operator-() const noexcept
	{
		return VectorNegate(*this);
	}

	inline MVector& __vectorcall MVector::operator+=(const MVector V) noexcept
	{
		this->v = _mm_add_ps(this->v, V.v);
		return *this;
	}

	inline MVector& __vectorcall MVector::operator-=(const MVector V) noexcept
	{
		this->v = _mm_sub_ps(this->v, V.v);
		return *this;
	}

	inline MVector& __vectorcall MVector::operator*=(const MVector V) noexcept
	{
		this->v = _mm_mul_ps(this->v, V.v);
		return *this;
	}

	inline MVector& __vectorcall MVector::operator/=(const MVector V) noexcept
	{
		this->v = _mm_div_ps(this->v, V.v);
		return *this;
	}

	inline MVector& MVector::operator*=(float S) noexcept
	{
		__m128 vS = _mm_set_ps1(S);
		this->v = _mm_mul_ps(this->v, vS);
		return *this;
	}

	inline MVector& MVector::operator/=(float S) noexcept
	{
		__m128 vS = _mm_set_ps1(S);
		this->v = _mm_div_ps(this->v, vS);
		return *this;
	}

	inline MVector __vectorcall MVector::operator+(const MVector V) const noexcept
	{
		return VectorAdd(*this, V);
	}

	inline MVector __vectorcall MVector::operator-(const MVector V) const noexcept
	{
		return VectorSub(*this, V);
	}

	inline MVector __vectorcall MVector::operator*(const MVector V) const noexcept
	{
		return VectorMul(*this, V);
	}

	inline MVector __vectorcall MVector::operator/(const MVector V) const noexcept
	{
		return VectorDiv(*this, V);
	}

	inline MVector MVector::operator*(float S) const noexcept
	{
		return VectorScale(*this, S);
	}

	inline MVector MVector::operator/(float S) const noexcept
	{
		MVector vRes;
		__m128 vS = _mm_set_ps1(S);
		vRes.v = _mm_div_ps(this->v, vS);
		return vRes;
	}

	inline MVector __vectorcall operator*(float S, const MVector V) noexcept
	{
		return VectorScale(V, S);
	}

	// Quaternion Operations
	inline MQuaternion __vectorcall QuaternionMul(const MQuaternion Q1, const MQuaternion Q2)
	{
		__m128 SignWZYX = _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f);
		__m128 SignZWXY = _mm_set_ps(-1.0f, -1.0f, 1.0f, 1.0f);
		__m128 SignYXWZ = _mm_set_ps(-1.0f, 1.0f, 1.0f, -1.0f);

		__m128 Q1X = _mm_shuffle_ps(Q1.q, Q1.q, _MM_SHUFFLE(0, 0, 0, 0));
		__m128 Q1Y = _mm_shuffle_ps(Q1.q, Q1.q, _MM_SHUFFLE(1, 1, 1, 1));
		__m128 Q1Z = _mm_shuffle_ps(Q1.q, Q1.q, _MM_SHUFFLE(2, 2, 2, 2));
		__m128 Q1W = _mm_shuffle_ps(Q1.q, Q1.q, _MM_SHUFFLE(3, 3, 3, 3));

		Q1W = _mm_mul_ps(Q1W, Q2.q);

		__m128 Q2Shuffle = _mm_shuffle_ps(Q2.q, Q2.q, _MM_SHUFFLE(0, 1, 2, 3));
		Q1X = _mm_mul_ps(SignWZYX, _mm_mul_ps(Q1X, Q2Shuffle));
		
		Q2Shuffle = _mm_shuffle_ps(Q2.q, Q2.q, _MM_SHUFFLE(1, 0, 3, 2));
		Q1Y = _mm_mul_ps(SignZWXY, _mm_mul_ps(Q1Y, Q2Shuffle));

		Q2Shuffle = _mm_shuffle_ps(Q2.q, Q2.q, _MM_SHUFFLE(2, 3, 0, 1));
		Q1Z = _mm_mul_ps(SignYXWZ, _mm_mul_ps(Q1Z, Q2Shuffle));

		MQuaternion qRes;
		qRes.q = _mm_add_ps(Q1W, _mm_add_ps(Q1X, _mm_add_ps(Q1Y, Q1Z)));
		return qRes;
	}

	inline MMatrix __vectorcall QuaternionToMatrix(const MQuaternion Q)
	{
		__m128 Constant1110 = _mm_set_ps(0.0f, 1.0f, 1.0f, 1.0f);

		__m128 Q0 = _mm_add_ps(Q.q, Q.q);
		__m128 Q1 = _mm_mul_ps(Q.q, Q0);

		__m128 V0 = _mm_shuffle_ps(Q1, Q1, _MM_SHUFFLE(3, 0, 0, 1));
		V0 = _mm_and_ps(V0, g_VecMaskXYZ);
		__m128 V1 = _mm_shuffle_ps(Q1, Q1, _MM_SHUFFLE(3, 1, 2, 2));
		V1 = _mm_and_ps(V1, g_VecMaskXYZ);
		__m128 R0 = _mm_sub_ps(Constant1110, V0);
		R0 = _mm_sub_ps(R0, V1);

		V0 = _mm_shuffle_ps(Q.q, Q.q, _MM_SHUFFLE(3, 1, 0, 0));
		V1 = _mm_shuffle_ps(Q0, Q0, _MM_SHUFFLE(3, 2, 1, 2));
		V0 = _mm_mul_ps(V0, V1);

		V1 = _mm_shuffle_ps(Q.q, Q.q, _MM_SHUFFLE(3, 3, 3, 3));
		__m128 V2 = _mm_shuffle_ps(Q0, Q0, _MM_SHUFFLE(3, 0, 2, 1));
		V1 = _mm_mul_ps(V1, V2);

		__m128 R1 = _mm_add_ps(V0, V1);
		__m128 R2 = _mm_sub_ps(V0, V1);

		V0 = _mm_shuffle_ps(R1, R2, _MM_SHUFFLE(1, 0, 2, 1));
		V0 = _mm_shuffle_ps(V0, V0, _MM_SHUFFLE(1, 3, 2, 0));
		V1 = _mm_shuffle_ps(R1, R2, _MM_SHUFFLE(2, 2, 0, 0));
		V1 = _mm_shuffle_ps(V1, V1, _MM_SHUFFLE(2, 0, 2, 0));

		Q1 = _mm_shuffle_ps(R0, V0, _MM_SHUFFLE(1, 0, 3, 0));
		Q1 = _mm_shuffle_ps(Q1, Q1, _MM_SHUFFLE(1, 3, 2, 0));

		MMatrix M;
		M.r[0] = Q1;

		Q1 = _mm_shuffle_ps(R0, V0, _MM_SHUFFLE(3, 2, 3, 1));
		Q1 = _mm_shuffle_ps(Q1, Q1, _MM_SHUFFLE(1, 3, 0, 2));
		M.r[1] = Q1;

		Q1 = _mm_shuffle_ps(V1, R0, _MM_SHUFFLE(3, 2, 1, 0));
		M.r[2] = Q1;
		M.r[3] = g_MatIdentityR3;
		return M;
	}

	// Quaternion Overloads
	inline MQuaternion MQuaternion::operator-() const noexcept
	{
		MQuaternion qRes;
		__m128 Z = _mm_setzero_ps();
		qRes.q = _mm_sub_ps(Z, this->q);
		return qRes;
	}

	inline MQuaternion& __vectorcall MQuaternion::operator+=(const MQuaternion Q) noexcept
	{
		this->q = _mm_add_ps(this->q, Q.q);
		return *this;
	}

	inline MQuaternion& __vectorcall MQuaternion::operator-=(const MQuaternion Q) noexcept
	{
		this->q = _mm_sub_ps(this->q, Q.q);
		return *this;
	}

	inline MQuaternion& __vectorcall MQuaternion::operator*=(const MQuaternion Q) noexcept
	{
		*this = QuaternionMul(*this, Q);
		return *this;
	}

	inline MQuaternion& MQuaternion::operator*=(float S) noexcept
	{
		__m128 vS = _mm_set_ps1(S);
		this->q = _mm_mul_ps(this->q, vS);
		return *this;
	}

	inline MQuaternion& MQuaternion::operator/=(float S) noexcept
	{
		__m128 vS = _mm_set_ps1(S);
		this->q = _mm_div_ps(this->q, vS);
		return *this;
	}

	inline MQuaternion __vectorcall MQuaternion::operator+(const MQuaternion Q) const noexcept
	{
		MQuaternion qRes;
		qRes.q = _mm_add_ps(this->q, Q.q);
		return qRes;
	}

	inline MQuaternion __vectorcall MQuaternion::operator-(const MQuaternion Q) const noexcept
	{
		MQuaternion qRes;
		qRes.q = _mm_sub_ps(this->q, Q.q);
		return qRes;
	}

	inline MQuaternion __vectorcall MQuaternion::operator*(const MQuaternion Q) const noexcept
	{
		return QuaternionMul(*this, Q);
	}

	inline MQuaternion MQuaternion::operator*(float S) const noexcept
	{
		__m128 vS = _mm_set_ps1(S);
		MQuaternion qRes;
		qRes.q = _mm_mul_ps(this->q, vS);
		return qRes;
	}

	inline MQuaternion MQuaternion::operator/(float S) const noexcept
	{
		__m128 vS = _mm_set_ps1(S);
		MQuaternion qRes;
		qRes.q = _mm_div_ps(this->q, vS);
		return qRes;
	}

	inline MQuaternion __vectorcall operator*(float S, const MQuaternion Q) noexcept
	{
		__m128 vS = _mm_set_ps1(S);
		MQuaternion qRes;
		qRes.q = _mm_mul_ps(vS, Q.q);
		return qRes;
	}

	// Matrix Operation
	inline MMatrix __vectorcall MatrixTranspose(const MMatrix M) noexcept
	{
		MMatrix mRes;

		__m128 vTemp1 = _mm_shuffle_ps(M.r[0], M.r[1], _MM_SHUFFLE(1, 0, 1, 0));
		__m128 vTemp3 = _mm_shuffle_ps(M.r[0], M.r[1], _MM_SHUFFLE(3, 2, 3, 2));
		__m128 vTemp2 = _mm_shuffle_ps(M.r[2], M.r[3], _MM_SHUFFLE(1, 0, 1, 0));
		__m128 vTemp4 = _mm_shuffle_ps(M.r[2], M.r[3], _MM_SHUFFLE(3, 2, 3, 2));

		mRes.r[0] = _mm_shuffle_ps(vTemp1, vTemp2, _MM_SHUFFLE(2, 0, 2, 0));
		mRes.r[1] = _mm_shuffle_ps(vTemp1, vTemp2, _MM_SHUFFLE(3, 1, 3, 1));
		mRes.r[2] = _mm_shuffle_ps(vTemp3, vTemp4, _MM_SHUFFLE(2, 0, 2, 0));
		mRes.r[3] = _mm_shuffle_ps(vTemp3, vTemp4, _MM_SHUFFLE(3, 1, 3, 1));
		return mRes;
	}

	inline MMatrix __vectorcall MatrixInverse(const MMatrix M) noexcept
	{
		// Transpose matrix
		__m128 vTemp1 = _mm_shuffle_ps(M.r[0], M.r[1], _MM_SHUFFLE(1, 0, 1, 0));
		__m128 vTemp3 = _mm_shuffle_ps(M.r[0], M.r[1], _MM_SHUFFLE(3, 2, 3, 2));
		__m128 vTemp2 = _mm_shuffle_ps(M.r[2], M.r[3], _MM_SHUFFLE(1, 0, 1, 0));
		__m128 vTemp4 = _mm_shuffle_ps(M.r[2], M.r[3], _MM_SHUFFLE(3, 2, 3, 2));

		MMatrix MT;
		MT.r[0] = _mm_shuffle_ps(vTemp1, vTemp2, _MM_SHUFFLE(2, 0, 2, 0));
		MT.r[1] = _mm_shuffle_ps(vTemp1, vTemp2, _MM_SHUFFLE(3, 1, 3, 1));
		MT.r[2] = _mm_shuffle_ps(vTemp3, vTemp4, _MM_SHUFFLE(2, 0, 2, 0));
		MT.r[3] = _mm_shuffle_ps(vTemp3, vTemp4, _MM_SHUFFLE(3, 1, 3, 1));

		__m128 V00 = _mm_shuffle_ps(MT.r[2], MT.r[2], _MM_SHUFFLE(1, 1, 0, 0));
		__m128 V10 = _mm_shuffle_ps(MT.r[3], MT.r[3], _MM_SHUFFLE(3, 2, 3, 2));
		__m128 V01 = _mm_shuffle_ps(MT.r[0], MT.r[0], _MM_SHUFFLE(1, 1, 0, 0));
		__m128 V11 = _mm_shuffle_ps(MT.r[1], MT.r[1], _MM_SHUFFLE(3, 2, 3, 2));
		__m128 V02 = _mm_shuffle_ps(MT.r[2], MT.r[0], _MM_SHUFFLE(2, 0, 2, 0));
		__m128 V12 = _mm_shuffle_ps(MT.r[3], MT.r[1], _MM_SHUFFLE(3, 1, 3, 1));

		__m128 D0 = _mm_mul_ps(V00, V10);
		__m128 D1 = _mm_mul_ps(V01, V11);
		__m128 D2 = _mm_mul_ps(V02, V12);

		V00 = _mm_shuffle_ps(MT.r[2], MT.r[2], _MM_SHUFFLE(3, 2, 3, 2));
		V10 = _mm_shuffle_ps(MT.r[3], MT.r[3], _MM_SHUFFLE(1, 1, 0, 0));
		V01 = _mm_shuffle_ps(MT.r[0], MT.r[0], _MM_SHUFFLE(3, 2, 3, 2));
		V11 = _mm_shuffle_ps(MT.r[1], MT.r[1], _MM_SHUFFLE(1, 1, 0, 0));
		V02 = _mm_shuffle_ps(MT.r[2], MT.r[0], _MM_SHUFFLE(3, 1, 3, 1));
		V12 = _mm_shuffle_ps(MT.r[3], MT.r[1], _MM_SHUFFLE(2, 0, 2, 0));

		D0 = _mm_sub_ps(D0, _mm_mul_ps(V00, V10));
		D1 = _mm_sub_ps(D1, _mm_mul_ps(V01, V11));
		D2 = _mm_sub_ps(D2, _mm_mul_ps(V02, V12));
		// V11 = D0Y,D0W,D2Y,D2Y
		V11 = _mm_shuffle_ps(D0, D2, _MM_SHUFFLE(1, 1, 3, 1));
		V00 = _mm_shuffle_ps(MT.r[1], MT.r[1], _MM_SHUFFLE(1, 0, 2, 1));
		V10 = _mm_shuffle_ps(V11, D0, _MM_SHUFFLE(0, 3, 0, 2));
		V01 = _mm_shuffle_ps(MT.r[0], MT.r[0], _MM_SHUFFLE(0, 1, 0, 2));
		V11 = _mm_shuffle_ps(V11, D0, _MM_SHUFFLE(2, 1, 2, 1));
		// V13 = D1Y,D1W,D2W,D2W
		__m128 V13 = _mm_shuffle_ps(D1, D2, _MM_SHUFFLE(3, 3, 3, 1));
		V02 = _mm_shuffle_ps(MT.r[3], MT.r[3], _MM_SHUFFLE(1, 0, 2, 1));
		V12 = _mm_shuffle_ps(V13, D1, _MM_SHUFFLE(0, 3, 0, 2));
		__m128 V03 = _mm_shuffle_ps(MT.r[2], MT.r[2], _MM_SHUFFLE(0, 1, 0, 2));
		V13 = _mm_shuffle_ps(V13, D1, _MM_SHUFFLE(2, 1, 2, 1));

		__m128 C0 = _mm_mul_ps(V00, V10);
		__m128 C2 = _mm_mul_ps(V01, V11);
		__m128 C4 = _mm_mul_ps(V02, V12);
		__m128 C6 = _mm_mul_ps(V03, V13);

		// V11 = D0X,D0Y,D2X,D2X
		V11 = _mm_shuffle_ps(D0, D2, _MM_SHUFFLE(0, 0, 1, 0));
		V00 = _mm_shuffle_ps(MT.r[1], MT.r[1], _MM_SHUFFLE(2, 1, 3, 2));
		V10 = _mm_shuffle_ps(D0, V11, _MM_SHUFFLE(2, 1, 0, 3));
		V01 = _mm_shuffle_ps(MT.r[0], MT.r[0], _MM_SHUFFLE(1, 3, 2, 3));
		V11 = _mm_shuffle_ps(D0, V11, _MM_SHUFFLE(0, 2, 1, 2));
		// V13 = D1X,D1Y,D2Z,D2Z
		V13 = _mm_shuffle_ps(D1, D2, _MM_SHUFFLE(2, 2, 1, 0));
		V02 = _mm_shuffle_ps(MT.r[3], MT.r[3], _MM_SHUFFLE(2, 1, 3, 2));
		V12 = _mm_shuffle_ps(D1, V13, _MM_SHUFFLE(2, 1, 0, 3));
		V03 = _mm_shuffle_ps(MT.r[2], MT.r[2], _MM_SHUFFLE(1, 3, 2, 3));
		V13 = _mm_shuffle_ps(D1, V13, _MM_SHUFFLE(0, 2, 1, 2));

		C0 = _mm_sub_ps(C0, _mm_mul_ps(V00, V10)); 
		C2 = _mm_sub_ps(C2, _mm_mul_ps(V01, V11));
		C4 = _mm_sub_ps(C4, _mm_mul_ps(V02, V12));
		C6 = _mm_sub_ps(C6, _mm_mul_ps(V03, V13));

		V00 = _mm_shuffle_ps(MT.r[1], MT.r[1], _MM_SHUFFLE(0, 3, 0, 3));
		// V10 = D0Z,D0Z,D2X,D2Y
		V10 = _mm_shuffle_ps(D0, D2, _MM_SHUFFLE(1, 0, 2, 2));
		V10 = _mm_shuffle_ps(V10, V10, _MM_SHUFFLE(0, 2, 3, 0));
		V01 = _mm_shuffle_ps(MT.r[0], MT.r[0], _MM_SHUFFLE(2, 0, 3, 1));
		// V11 = D0X,D0W,D2X,D2Y
		V11 = _mm_shuffle_ps(D0, D2, _MM_SHUFFLE(1, 0, 3, 0));
		V11 = _mm_shuffle_ps(V11, V11, _MM_SHUFFLE(2, 1, 0, 3));
		V02 = _mm_shuffle_ps(MT.r[3], MT.r[3], _MM_SHUFFLE(0, 3, 0, 3));
		// V12 = D1Z,D1Z,D2Z,D2W
		V12 = _mm_shuffle_ps(D1, D2, _MM_SHUFFLE(3, 2, 2, 2));
		V12 = _mm_shuffle_ps(V12, V12, _MM_SHUFFLE(0, 2, 3, 0));
		V03 = _mm_shuffle_ps(MT.r[2], MT.r[2], _MM_SHUFFLE(2, 0, 3, 1));
		// V13 = D1X,D1W,D2Z,D2W
		V13 = _mm_shuffle_ps(D1, D2, _MM_SHUFFLE(3, 2, 3, 0));
		V13 = _mm_shuffle_ps(V13, V13, _MM_SHUFFLE(2, 1, 0, 3));

		V00 = _mm_mul_ps(V00, V10);
		V01 = _mm_mul_ps(V01, V11);
		V02 = _mm_mul_ps(V02, V12);
		V03 = _mm_mul_ps(V03, V13);
		__m128 C1 = _mm_sub_ps(C0, V00);
		C0 = _mm_add_ps(C0, V00);
		__m128 C3 = _mm_add_ps(C2, V01);
		C2 = _mm_sub_ps(C2, V01);
		__m128 C5 = _mm_sub_ps(C4, V02);
		C4 = _mm_add_ps(C4, V02);
		__m128 C7 = _mm_add_ps(C6, V03);
		C6 = _mm_sub_ps(C6, V03);

		C0 = _mm_shuffle_ps(C0, C1, _MM_SHUFFLE(3, 1, 2, 0));
		C2 = _mm_shuffle_ps(C2, C3, _MM_SHUFFLE(3, 1, 2, 0));
		C4 = _mm_shuffle_ps(C4, C5, _MM_SHUFFLE(3, 1, 2, 0));
		C6 = _mm_shuffle_ps(C6, C7, _MM_SHUFFLE(3, 1, 2, 0));
		C0 = _mm_shuffle_ps(C0, C0, _MM_SHUFFLE(3, 1, 2, 0));
		C2 = _mm_shuffle_ps(C2, C2, _MM_SHUFFLE(3, 1, 2, 0));
		C4 = _mm_shuffle_ps(C4, C4, _MM_SHUFFLE(3, 1, 2, 0));
		C6 = _mm_shuffle_ps(C6, C6, _MM_SHUFFLE(3, 1, 2, 0));

		// Get the determinant
		__m128 vTemp = _mm_dp_ps(C0, MT.r[0], 0xff);
		__m128 vOne = _mm_set_ps1(1.0f);
		vTemp = _mm_div_ps(vOne, vTemp);
		MMatrix mResult;
		mResult.r[0] = _mm_mul_ps(C0, vTemp);
		mResult.r[1] = _mm_mul_ps(C2, vTemp);
		mResult.r[2] = _mm_mul_ps(C4, vTemp);
		mResult.r[3] = _mm_mul_ps(C6, vTemp);
		return mResult;
	}

	inline MQuaternion __vectorcall MatrixToQuaternion(const MMatrix M) noexcept
	{
		__m128 XMPMMP = _mm_set_ps(1.0f, -1.0f, -1.0f, 1.0f);
		__m128 XMMPMP = _mm_set_ps(1.0f, -1.0f, 1.0f, -1.0f);
		__m128 XMMMPP = _mm_set_ps(1.0f, 1.0f, -1.0f, -1.0f);

		__m128 r0 = M.r[0];  // (r00, r01, r02, 0)
		__m128 r1 = M.r[1];  // (r10, r11, r12, 0)
		__m128 r2 = M.r[2];  // (r20, r21, r22, 0)

		// (r00, r00, r00, r00)
		__m128 r00 = _mm_shuffle_ps(r0, r0, _MM_SHUFFLE(0, 0, 0, 0));
		// (r11, r11, r11, r11)
		__m128 r11 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(1, 1, 1, 1));
		// (r22, r22, r22, r22)
		__m128 r22 = _mm_shuffle_ps(r2, r2, _MM_SHUFFLE(2, 2, 2, 2));

		__m128 vZ = _mm_setzero_ps();
		__m128 vOne = _mm_set_ps1(1.0f);
		// x^2 >= y^2 equivalent to r11 - r00 <= 0
		// (r11 - r00, r11 - r00, r11 - r00, r11 - r00)
		__m128 r11mr00 = _mm_sub_ps(r11, r00);
		__m128 x2gey2 = _mm_cmple_ps(r11mr00, vZ);

		// z^2 >= w^2 equivalent to r11 + r00 <= 0
		// (r11 + r00, r11 + r00, r11 + r00, r11 + r00)
		__m128 r11pr00 = _mm_add_ps(r11, r00);
		__m128 z2gew2 = _mm_cmple_ps(r11pr00, vZ);

		// x^2 + y^2 >= z^2 + w^2 equivalent to r22 <= 0
		__m128 x2py2gez2pw2 = _mm_cmple_ps(r22, vZ);

		// (4*x^2, 4*y^2, 4*z^2, 4*w^2)
		__m128 t0 = _mm_add_ps(_mm_mul_ps(XMPMMP, r00), vOne); 
		__m128 t1 = _mm_mul_ps(XMMPMP, r11);
		__m128 t2 = _mm_add_ps(_mm_mul_ps(XMMMPP, r22), t0);
		__m128 x2y2z2w2 = _mm_add_ps(t1, t2);

		// (r01, r02, r12, r11)
		t0 = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(1, 2, 2, 1));
		// (r10, r10, r20, r21)
		t1 = _mm_shuffle_ps(r1, r2, _MM_SHUFFLE(1, 0, 0, 0));
		// (r10, r20, r21, r10)
		t1 = _mm_shuffle_ps(t1, t1, _MM_SHUFFLE(1, 3, 2, 0));
		// (4*x*y, 4*x*z, 4*y*z, unused)
		__m128 xyxzyz = _mm_add_ps(t0, t1);

		// (r21, r20, r10, r10)
		t0 = _mm_shuffle_ps(r2, r1, _MM_SHUFFLE(0, 0, 0, 1));
		// (r12, r12, r02, r01)
		t1 = _mm_shuffle_ps(r1, r0, _MM_SHUFFLE(1, 2, 2, 2));
		// (r12, r02, r01, r12)
		t1 = _mm_shuffle_ps(t1, t1, _MM_SHUFFLE(1, 3, 2, 0));
		// (4*x*w, 4*y*w, 4*z*w, unused)
		__m128 xwywzw = _mm_sub_ps(t0, t1);
		xwywzw = _mm_mul_ps(XMMPMP, xwywzw);

		// (4*x^2, 4*y^2, 4*x*y, unused)
		t0 = _mm_shuffle_ps(x2y2z2w2, xyxzyz, _MM_SHUFFLE(0, 0, 1, 0));
		// (4*z^2, 4*w^2, 4*z*w, unused)
		t1 = _mm_shuffle_ps(x2y2z2w2, xwywzw, _MM_SHUFFLE(0, 2, 3, 2));
		// (4*x*z, 4*y*z, 4*x*w, 4*y*w)
		t2 = _mm_shuffle_ps(xyxzyz, xwywzw, _MM_SHUFFLE(1, 0, 2, 1));

		// (4*x*x, 4*x*y, 4*x*z, 4*x*w)
		__m128 tensor0 = _mm_shuffle_ps(t0, t2, _MM_SHUFFLE(2, 0, 2, 0));
		// (4*y*x, 4*y*y, 4*y*z, 4*y*w)
		__m128 tensor1 = _mm_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 1, 1, 2));
		// (4*z*x, 4*z*y, 4*z*z, 4*z*w)
		__m128 tensor2 = _mm_shuffle_ps(t2, t1, _MM_SHUFFLE(2, 0, 1, 0));
		// (4*w*x, 4*w*y, 4*w*z, 4*w*w)
		__m128 tensor3 = _mm_shuffle_ps(t2, t1, _MM_SHUFFLE(1, 2, 3, 2));

		// Select the row of the tensor-product matrix that has the largest
		// magnitude.
		t0 = _mm_and_ps(x2gey2, tensor0);
		t1 = _mm_andnot_ps(x2gey2, tensor1);
		t0 = _mm_or_ps(t0, t1);
		t1 = _mm_and_ps(z2gew2, tensor2);
		t2 = _mm_andnot_ps(z2gew2, tensor3);
		t1 = _mm_or_ps(t1, t2);
		t0 = _mm_and_ps(x2py2gez2pw2, t0);
		t1 = _mm_andnot_ps(x2py2gez2pw2, t1);
		t2 = _mm_or_ps(t0, t1);

		// Normalize the row.  No division by zero is possible because the
		// quaternion is unit-length (and the row is a nonzero multiple of
		// the quaternion).
		MQuaternion qRes;
		t0 = _mm_sqrt_ps(_mm_dp_ps(t2, t2, 0xff));
		qRes.q = _mm_div_ps(t2, t0);
		return qRes;
	}

	inline MMatrix __vectorcall MatrixMul(const MMatrix M1, const MMatrix& M2) noexcept
	{
		MMatrix mRes;

		//__m128 vX = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[0]) + 0);
		//__m128 vY = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[0]) + 1);
		//__m128 vZ = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[0]) + 2);
		//__m128 vW = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[0]) + 3);

		__m128 vX = _mm_shuffle_ps(M2.r[0], M2.r[0], _MM_SHUFFLE(0, 0, 0, 0));
		__m128 vY = _mm_shuffle_ps(M2.r[0], M2.r[0], _MM_SHUFFLE(1, 1, 1, 1));
		__m128 vZ = _mm_shuffle_ps(M2.r[0], M2.r[0], _MM_SHUFFLE(2, 2, 2, 2));
		__m128 vW = _mm_shuffle_ps(M2.r[0], M2.r[0], _MM_SHUFFLE(3, 3, 3, 3));

		vX = _mm_mul_ps(vX, M1.r[0]);
		vY = _mm_mul_ps(vY, M1.r[1]);
		vZ = _mm_mul_ps(vZ, M1.r[2]);
		vW = _mm_mul_ps(vW, M1.r[3]);

		vX = _mm_add_ps(vX, vZ);
		vY = _mm_add_ps(vY, vW);
		vX = _mm_add_ps(vX, vY);
		mRes.r[0] = vX;

		//vX = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[1]) + 0);
		//vY = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[1]) + 1);
		//vZ = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[1]) + 2);
		//vW = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[1]) + 3);

		vX = _mm_shuffle_ps(M2.r[1], M2.r[1], _MM_SHUFFLE(0, 0, 0, 0));
		vY = _mm_shuffle_ps(M2.r[1], M2.r[1], _MM_SHUFFLE(1, 1, 1, 1));
		vZ = _mm_shuffle_ps(M2.r[1], M2.r[1], _MM_SHUFFLE(2, 2, 2, 2));
		vW = _mm_shuffle_ps(M2.r[1], M2.r[1], _MM_SHUFFLE(3, 3, 3, 3));

		vX = _mm_mul_ps(vX, M1.r[0]);
		vY = _mm_mul_ps(vY, M1.r[1]);
		vZ = _mm_mul_ps(vZ, M1.r[2]);
		vW = _mm_mul_ps(vW, M1.r[3]);

		vX = _mm_add_ps(vX, vZ);
		vY = _mm_add_ps(vY, vW);
		vX = _mm_add_ps(vX, vY);
		mRes.r[1] = vX;

		//vX = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[2]) + 0);
		//vY = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[2]) + 1);
		//vZ = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[2]) + 2);
		//vW = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[2]) + 3);

		vX = _mm_shuffle_ps(M2.r[2], M2.r[2], _MM_SHUFFLE(0, 0, 0, 0));
		vY = _mm_shuffle_ps(M2.r[2], M2.r[2], _MM_SHUFFLE(1, 1, 1, 1));
		vZ = _mm_shuffle_ps(M2.r[2], M2.r[2], _MM_SHUFFLE(2, 2, 2, 2));
		vW = _mm_shuffle_ps(M2.r[2], M2.r[2], _MM_SHUFFLE(3, 3, 3, 3));

		vX = _mm_mul_ps(vX, M1.r[0]);
		vY = _mm_mul_ps(vY, M1.r[1]);
		vZ = _mm_mul_ps(vZ, M1.r[2]);
		vW = _mm_mul_ps(vW, M1.r[3]);

		vX = _mm_add_ps(vX, vZ);
		vY = _mm_add_ps(vY, vW);
		vX = _mm_add_ps(vX, vY);
		mRes.r[2] = vX;

		//vX = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[3]) + 0);
		//vY = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[3]) + 1);
		//vZ = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[3]) + 2);
		//vW = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[3]) + 3);

		vX = _mm_shuffle_ps(M2.r[3], M2.r[3], _MM_SHUFFLE(0, 0, 0, 0));
		vY = _mm_shuffle_ps(M2.r[3], M2.r[3], _MM_SHUFFLE(1, 1, 1, 1));
		vZ = _mm_shuffle_ps(M2.r[3], M2.r[3], _MM_SHUFFLE(2, 2, 2, 2));
		vW = _mm_shuffle_ps(M2.r[3], M2.r[3], _MM_SHUFFLE(3, 3, 3, 3));

		vX = _mm_mul_ps(vX, M1.r[0]);
		vY = _mm_mul_ps(vY, M1.r[1]);
		vZ = _mm_mul_ps(vZ, M1.r[2]);
		vW = _mm_mul_ps(vW, M1.r[3]);

		vX = _mm_add_ps(vX, vZ);
		vY = _mm_add_ps(vY, vW);
		vX = _mm_add_ps(vX, vY);
		mRes.r[3] = vX;
		return mRes;
	}

	inline MMatrix __vectorcall MatrixLookAtRH(const MVector EyePos, const MVector FocusPos, const MVector UpDir) noexcept
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

		M = MatrixTranspose(M);
		return M;
	}

	inline MMatrix __vectorcall MatrixPerspectiveFovRH(float FovY, float AspectRation, float NearZ, float FarZ) noexcept
	{
		float fRange = FarZ / (NearZ - FarZ);
		float Height = ScalarCos(FovY / 2) / ScalarSin(FovY / 2);
		__m128 V = _mm_set_ps(fRange * NearZ, fRange, Height, Height / AspectRation);
		__m128 vTemp = _mm_set_ps(-1.0f, 0.0f, 0.0f, 0.0f);

		MMatrix M;
		M.r[0] = _mm_and_ps(V, g_VecMaskX);
		M.r[1] = _mm_and_ps(V, g_VecMaskY);
		V = _mm_shuffle_ps(V, vTemp, _MM_SHUFFLE(3, 2, 3, 2));
		vTemp = _mm_setzero_ps();
		vTemp = _mm_shuffle_ps(vTemp, V, _MM_SHUFFLE(3, 0, 0, 0));
		M.r[2] = vTemp;
		vTemp = _mm_shuffle_ps(vTemp, V, _MM_SHUFFLE(2, 1, 0, 0));
		M.r[3] = vTemp;
		return M;
	}

	inline MMatrix __vectorcall MatrixOrthographicRH(float ViewWidth, float ViewHeight, float NearZ, float FarZ) noexcept
	{
		float fRange = 1.0f / (NearZ - FarZ);
		__m128 V = _mm_set_ps(fRange * NearZ, fRange, 2.0f / ViewHeight, 2.0f / ViewWidth);
		__m128 vTemp = _mm_setzero_ps();

		MMatrix M;
		M.r[0] = _mm_and_ps(V, g_VecMaskX);
		M.r[1] = _mm_and_ps(V, g_VecMaskY);
		V = _mm_shuffle_ps(V, g_MatIdentityR3, _MM_SHUFFLE(3, 2, 3, 2));
		vTemp = _mm_shuffle_ps(vTemp, V, _MM_SHUFFLE(2, 0, 0, 0));
		M.r[2] = vTemp;
		vTemp = _mm_shuffle_ps(vTemp, V, _MM_SHUFFLE(3, 0, 0, 0));
		M.r[3] = vTemp;
		return M;
	}

	// Matrix Overloads
	inline MMatrix MMatrix::operator-() const noexcept
	{
		MMatrix M;
		__m128 Z = _mm_setzero_ps();
		M.r[0] = _mm_sub_ps(Z, this->r[0]);
		M.r[1] = _mm_sub_ps(Z, this->r[1]);
		M.r[2] = _mm_sub_ps(Z, this->r[2]);
		M.r[3] = _mm_sub_ps(Z, this->r[3]);
		return M;
	}

	inline MMatrix& __vectorcall MMatrix::operator+=(const MMatrix M) noexcept
	{
		this->r[0] = _mm_add_ps(this->r[0], M.r[0]);
		this->r[1] = _mm_add_ps(this->r[0], M.r[1]);
		this->r[2] = _mm_add_ps(this->r[0], M.r[2]);
		this->r[3] = _mm_add_ps(this->r[0], M.r[3]);
		return *this;
	}

	inline MMatrix& __vectorcall MMatrix::operator-=(const MMatrix M) noexcept
	{
		this->r[0] = _mm_sub_ps(this->r[0], M.r[0]);
		this->r[1] = _mm_sub_ps(this->r[0], M.r[1]);
		this->r[2] = _mm_sub_ps(this->r[0], M.r[2]);
		this->r[3] = _mm_sub_ps(this->r[0], M.r[3]);
		return *this;
	}

	inline MMatrix& __vectorcall MMatrix::operator*=(const MMatrix M) noexcept
	{
		*this = MatrixMul(*this, M);
		return *this;
	}

	inline MMatrix& MMatrix::operator*=(float S) noexcept
	{
		__m128 vS = _mm_set_ps1(S);
		this->r[0] = _mm_mul_ps(this->r[0], vS);
		this->r[1] = _mm_mul_ps(this->r[1], vS);
		this->r[2] = _mm_mul_ps(this->r[2], vS);
		this->r[3] = _mm_mul_ps(this->r[3], vS);
		return *this;
	}

	inline MMatrix& MMatrix::operator/=(float S) noexcept
	{
		__m128 vS = _mm_set_ps1(S);
		this->r[0] = _mm_div_ps(this->r[0], vS);
		this->r[1] = _mm_div_ps(this->r[1], vS);
		this->r[2] = _mm_div_ps(this->r[2], vS);
		this->r[3] = _mm_div_ps(this->r[3], vS);
		return *this;
	}

	inline MMatrix __vectorcall MMatrix::operator+(const MMatrix M) const noexcept
	{
		MMatrix mRes;
		mRes.r[0] = _mm_add_ps(this->r[0], M.r[0]);
		mRes.r[1] = _mm_add_ps(this->r[1], M.r[1]);
		mRes.r[2] = _mm_add_ps(this->r[2], M.r[2]);
		mRes.r[3] = _mm_add_ps(this->r[3], M.r[3]);
		return mRes;
	}

	inline MMatrix __vectorcall MMatrix::operator-(const MMatrix M) const noexcept
	{
		MMatrix mRes;
		mRes.r[0] = _mm_sub_ps(this->r[0], M.r[0]);
		mRes.r[1] = _mm_sub_ps(this->r[1], M.r[1]);
		mRes.r[2] = _mm_sub_ps(this->r[2], M.r[2]);
		mRes.r[3] = _mm_sub_ps(this->r[3], M.r[3]);
		return mRes;
	}

	inline MMatrix __vectorcall MMatrix::operator*(const MMatrix M) const noexcept
	{
		return MatrixMul(*this, M);
	}

	inline MMatrix MMatrix::operator*(float S) const noexcept
	{
		MMatrix mRes;
		__m128 vS = _mm_set_ps1(S);
		mRes.r[0] = _mm_mul_ps(this->r[0], vS);
		mRes.r[1] = _mm_mul_ps(this->r[1], vS);
		mRes.r[2] = _mm_mul_ps(this->r[2], vS);
		mRes.r[3] = _mm_mul_ps(this->r[3], vS);
		return mRes;
	}

	inline MMatrix MMatrix::operator/(float S) const noexcept
	{
		MMatrix mRes;
		__m128 vS = _mm_set_ps1(S);
		mRes.r[0] = _mm_div_ps(this->r[0], vS);
		mRes.r[1] = _mm_div_ps(this->r[1], vS);
		mRes.r[2] = _mm_div_ps(this->r[2], vS);
		mRes.r[3] = _mm_div_ps(this->r[3], vS);
		return mRes;
	}

	inline MMatrix __vectorcall operator*(float S, const MMatrix M) noexcept
	{
		MMatrix mRes;
		__m128 vS = _mm_set_ps1(S);
		mRes.r[0] = _mm_mul_ps(M.r[0], vS);
		mRes.r[1] = _mm_mul_ps(M.r[1], vS);
		mRes.r[2] = _mm_mul_ps(M.r[2], vS);
		mRes.r[3] = _mm_mul_ps(M.r[3], vS);
		return mRes;
	}

	inline MVector __vectorcall MMatrix::operator*(const MVector V) const noexcept
	{
		return MatrixMulVec(*this, V);
	}

	//#include "GMathVector.inl"
	//#include "GMathMatrix.inl"
	//#include "GMathQuaternion.inl"
}