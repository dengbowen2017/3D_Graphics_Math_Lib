#pragma once

#include <cassert>
#include <immintrin.h>

namespace GMath
{
	using MVector = __m128;

	struct MMatrix
	{
		MVector r[4];
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

	// Global values

	extern const __declspec(selectany) MVector g_MatIdentityR0 = _mm_set_ps(0, 0, 0, 1.0f);
	extern const __declspec(selectany) MVector g_MatIdentityR1 = _mm_set_ps(0, 0, 1.0f, 0);
	extern const __declspec(selectany) MVector g_MatIdentityR2 = _mm_set_ps(0, 1.0f, 0, 0);
	extern const __declspec(selectany) MVector g_MatIdentityR3 = _mm_set_ps(1.0f, 0, 0, 0);

																					   //w		   //z		   //y	       //x
	extern const __declspec(selectany) MVector g_VecMaskXYZ = *(__m128*)&_mm_set_epi32(0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);

	// Load Operations

	MVector __vectorcall LoadVector3(const Vector3* p_src) noexcept
	{
		assert(p_src);
		__m128 xy = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(p_src)));
		__m128 z = _mm_load_ss(&p_src->z);
		return _mm_movelh_ps(xy, z);
	}

	MVector __vectorcall LoadVector3A(const Vector3A* p_src) noexcept
	{
		assert(p_src);
		assert((reinterpret_cast<uintptr_t>(p_src) & 0xF) == 0);
		__m128 V = _mm_load_ps(&p_src->x);
		return _mm_and_ps(V, g_VecMaskXYZ);
	}

	MVector __vectorcall LoadVector4(const Vector4* p_src) noexcept
	{
		assert(p_src);
		return _mm_loadu_ps(&p_src->x);
	}

	MVector __vectorcall LoadVector4A(const Vector4A* p_src) noexcept
	{
		assert(p_src);
		assert((reinterpret_cast<uintptr_t>(p_src) & 0xF) == 0);
		return _mm_load_ps(&p_src->x);
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

	void __vectorcall StoreVector3(Vector3* p_des, const MVector V)
	{
		assert(p_des);
		_mm_store_sd(reinterpret_cast<double*>(p_des), _mm_castps_pd(V));
		__m128 z = _mm_shuffle_ps(V, V, _MM_SHUFFLE(2, 2, 2, 2));
		_mm_store_ss(&p_des->z, z);
	}

	void __vectorcall StoreVector3A(Vector3A* p_des, const MVector V)
	{
		assert(p_des);
		assert((reinterpret_cast<uintptr_t>(p_des) & 0xF) == 0);
		_mm_store_sd(reinterpret_cast<double*>(p_des), _mm_castps_pd(V));
		__m128 z = _mm_movehl_ps(V, V);
		_mm_store_ss(&p_des->z, z);
	}

	void __vectorcall StoreVector4(Vector4* p_des, const MVector V)
	{
		assert(p_des);
		_mm_storeu_ps(&p_des->x, V);
	}

	void __vectorcall StoreVector4A(Vector4A* p_des, const MVector V)
	{
		assert(p_des);
		assert((reinterpret_cast<uintptr_t>(p_des) & 0xF) == 0);
		_mm_store_ps(&p_des->x, V);
	}

	void __vectorcall StoreMatrix3x3(Matrix3x3* p_des, const MMatrix M)
	{
		assert(p_des);
		MVector vTemp1 = M.r[0];
		MVector vTemp2 = M.r[1];
		MVector vTemp3 = M.r[2];
		MVector vWork = _mm_shuffle_ps(vTemp1, vTemp2, _MM_SHUFFLE(0, 0, 2, 2));
		vTemp1 = _mm_shuffle_ps(vTemp1, vWork, _MM_SHUFFLE(2, 0, 1, 0));
		_mm_storeu_ps(&p_des->m[0][0], vTemp1);
		vTemp2 = _mm_shuffle_ps(vTemp2, vTemp3, _MM_SHUFFLE(1, 0, 2, 1));
		_mm_storeu_ps(&p_des->m[1][1], vTemp2);
		vTemp3 = _mm_shuffle_ps(vTemp3, vTemp3, _MM_SHUFFLE(2, 2, 2, 2));
		_mm_store_ss(&p_des->m[2][2], vTemp3);
	}

	void __vectorcall StoreMatrix4x4(Matrix4x4* p_des, const MMatrix M)
	{
		assert(p_des);
		_mm_storeu_ps(&p_des->m[0][0], M.r[0]);
		_mm_storeu_ps(&p_des->m[1][0], M.r[1]);
		_mm_storeu_ps(&p_des->m[2][0], M.r[2]);
		_mm_storeu_ps(&p_des->m[3][0], M.r[3]);
	}

	void __vectorcall StoreMatrix4x4A(Matrix4x4A* p_des, const MMatrix M)
	{
		assert(p_des);
		assert((reinterpret_cast<uintptr_t>(p_des) & 0xF) == 0);
		_mm_store_ps(&p_des->m[0][0], M.r[0]);
		_mm_store_ps(&p_des->m[1][0], M.r[1]);
		_mm_store_ps(&p_des->m[2][0], M.r[2]);
		_mm_store_ps(&p_des->m[3][0], M.r[3]);
	}

}