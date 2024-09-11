#pragma once

#include <assert.h>
#include <immintrin.h>
#include <math.h>

namespace GMath
{
	// -------------------------
	// Constant Definitions
	// -------------------------

	constexpr float MATH_PI = 3.141592654f;

	constexpr float ConvertToRadians(float Degrees) noexcept { return Degrees * MATH_PI / 180.0f; }
	constexpr float ConvertToDegrees(float Radians) noexcept { return Radians * 180.0f / MATH_PI; }

	// -------------------------
	// Global Values
	// -------------------------

	extern const __declspec(selectany) __m128 g_VecAllZero = _mm_setzero_ps();
	extern const __declspec(selectany) __m128 g_VecAllOne = _mm_set_ps1(1.0f);

	extern const __declspec(selectany) __m128 g_MatIdentityR0 = _mm_set_ps(0, 0, 0, 1.0f);
	extern const __declspec(selectany) __m128 g_MatIdentityR1 = _mm_set_ps(0, 0, 1.0f, 0);
	extern const __declspec(selectany) __m128 g_MatIdentityR2 = _mm_set_ps(0, 1.0f, 0, 0);
	extern const __declspec(selectany) __m128 g_MatIdentityR3 = _mm_set_ps(1.0f, 0, 0, 0);

	extern const __declspec(selectany) __m128 g_VecMaskX = *(__m128*) & _mm_set_epi32(0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF); // wzyx
	extern const __declspec(selectany) __m128 g_VecMaskY = *(__m128*) & _mm_set_epi32(0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000);
	extern const __declspec(selectany) __m128 g_VecMaskZ = *(__m128*) & _mm_set_epi32(0x00000000, 0xFFFFFFFF, 0x00000000, 0x00000000);
	extern const __declspec(selectany) __m128 g_VecMaskW = *(__m128*) & _mm_set_epi32(0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000);
	extern const __declspec(selectany) __m128 g_VecMaskZW = *(__m128*) & _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000);
	extern const __declspec(selectany) __m128 g_VecMaskXYZ = *(__m128*) & _mm_set_epi32(0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);

	// --------------------------
	// Type Definitions
	// --------------------------

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
		MQuaternion(const MVector& V) noexcept : q(V.v) {}

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

		static MMatrix Zero() noexcept
		{
			MMatrix mRes;
			mRes.r[0] = g_VecAllZero;
			mRes.r[1] = g_VecAllZero;
			mRes.r[2] = g_VecAllZero;
			mRes.r[3] = g_VecAllZero;
			return mRes;
		}

		static MMatrix Identity() noexcept
		{
			MMatrix mRes;
			mRes.r[0] = g_MatIdentityR0;
			mRes.r[1] = g_MatIdentityR1;
			mRes.r[2] = g_MatIdentityR2;
			mRes.r[3] = g_MatIdentityR3;
			return mRes;
		}

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

	struct Vector2
	{
		float x;
		float y;

		Vector2() = default;
		constexpr Vector2(float _x, float _y) noexcept : x(_x), y(_y) {};
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

	// -----------------------------
	// Miscellaneous Operations
	// -----------------------------

	inline float ScalarSin(float Radians) noexcept { return sin(Radians); }
	inline float ScalarCos(float Radians) noexcept { return cos(Radians); }

	// --------------------------
	// Load Operations
	// --------------------------

	MVector __vectorcall LoadVector3(const Vector3* p_src) noexcept;
	MVector __vectorcall LoadVector3A(const Vector3A* p_src) noexcept;
	MVector __vectorcall LoadVector4(const Vector4* p_src) noexcept;
	MVector __vectorcall LoadVector4A(const Vector4A* p_src) noexcept;

	MMatrix __vectorcall LoadMatrix3x3(const Matrix3x3* p_src) noexcept;
	MMatrix __vectorcall LoadMatrix4x4(const Matrix4x4* p_src) noexcept;
	MMatrix __vectorcall LoadMatrix4x4A(const Matrix4x4A* p_src) noexcept;

	// -------------------------
	// Store Operations
	// -------------------------

	void __vectorcall StoreVector3(Vector3* p_des, const MVector V) noexcept;
	void __vectorcall StoreVector3A(Vector3A* p_des, const MVector V) noexcept;
	void __vectorcall StoreVector4(Vector4* p_des, const MVector V) noexcept;
	void __vectorcall StoreVector4A(Vector4A* p_des, const MVector V) noexcept;
	void __vectorcall StoreMatrix3x3(Matrix3x3* p_des, const MMatrix M) noexcept;
	void __vectorcall StoreMatrix4x4(Matrix4x4* p_des, const MMatrix M) noexcept;
	void __vectorcall StoreMatrix4x4A(Matrix4x4A* p_des, const MMatrix M) noexcept;

	// --------------------------
	// Vector Operations
	// --------------------------

	float __vectorcall VectorGetX(const MVector V) noexcept;

	MVector __vectorcall VectorNegate(const MVector V) noexcept;
	MVector __vectorcall VectorAdd(const MVector V1, const MVector V2) noexcept;
	MVector __vectorcall VectorSub(const MVector V1, const MVector V2) noexcept;
	MVector __vectorcall VectorMul(const MVector V1, const MVector V2) noexcept;
	MVector __vectorcall VectorDiv(const MVector V1, const MVector V2) noexcept;
	MVector __vectorcall VectorSum(const MVector V) noexcept;
	MVector __vectorcall VectorScale(const MVector V, float S) noexcept;
	MVector __vectorcall VectorMagnitude(const MVector V) noexcept;
	MVector __vectorcall VectorNormalize(const MVector V) noexcept;
	MVector __vectorcall VectorDot(const MVector V1, const MVector V2) noexcept;
	MVector __vectorcall VectorCross(const MVector V1, const MVector V2) noexcept;

	MVector __vectorcall MatrixMulVector(const MMatrix M, const MVector V) noexcept;
	MMatrix __vectorcall VectorCrossToMatrix(const MVector V) noexcept;


	// -----------------------------
	// Quaternion Operations
	// -----------------------------

	MQuaternion __vectorcall QuaternionMul(const MQuaternion Q1, const MQuaternion Q2) noexcept;
	MQuaternion __vectorcall QuaternionNormalize(const MQuaternion Q) noexcept;
	
	MMatrix __vectorcall QuaternionToMatrix(const MQuaternion Q) noexcept;

	// --------------------------
	// Matrix Operation
	// --------------------------

	MMatrix __vectorcall MatrixTranspose(const MMatrix M) noexcept;
	MMatrix __vectorcall MatrixInverse(const MMatrix M) noexcept;
	MMatrix __vectorcall MatrixMul(const MMatrix M1, const MMatrix& M2) noexcept;
	MMatrix __vectorcall MatrixLookAtRH(const MVector EyePos, const MVector FocusPos, const MVector UpDir) noexcept;
	MMatrix __vectorcall MatrixPerspectiveFovRH(float FovY, float AspectRation, float NearZ, float FarZ) noexcept;
	MMatrix __vectorcall MatrixOrthographicRH(float ViewWidth, float ViewHeight, float NearZ, float FarZ) noexcept;

	MQuaternion __vectorcall MatrixToQuaternion(const MMatrix M) noexcept;
	
	MMatrix __vectorcall TranslateMatrix(const MVector Pos) noexcept;
	MMatrix __vectorcall RotateMatrix(const MQuaternion Q) noexcept;
	MMatrix __vectorcall ScaleMatrix(const MVector Scale) noexcept;
	MMatrix __vectorcall ModelMatrix(const MVector Pos, const MQuaternion Q, const MVector Scale) noexcept;

	MMatrix __vectorcall PartOfInertiaMatrix(const MVector V) noexcept; // temp


#include "GMathConvert.inl"
#include "GMathVector.inl"
#include "GMathMatrix.inl"
#include "GMathQuaternion.inl"
}