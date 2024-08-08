#pragma once

#include <cassert>
#include <immintrin.h>

namespace GMath
{
	using MVector = __m128;
	typedef const MVector CMVector;

	struct MMatrix;
	typedef const MMatrix CMMatrix;

	struct MMatrix
	{
		MVector m[4];


	};

	struct Vector3
	{
		float x;
		float y;
		float z;

		constexpr Vector3(float init) noexcept : x(init), y(init), z(init) {};
		constexpr Vector3(float _x, float _y, float _z) noexcept : x(_x), y(_y), z(_z) {};
	};

	struct Vector4
	{
		float x;
		float y;
		float z;
		float w;

		constexpr Vector4(float init) noexcept : x(init), y(init), z(init), w(init) {};
		constexpr Vector4(float _x, float _y, float _z, float _w) noexcept : x(_x), y(_y), z(_z), w(_w) {};
	};

	struct Matrix3x3
	{
		float m[3][3];

		float operator() (unsigned int column, unsigned int row) const
		{
			assert(column < 3 && row < 3);
			return m[column][row];
		}

		float& operator() (unsigned int column, unsigned int row)
		{
			assert(column < 3 && row < 3);
			return m[column][row];
		}
	};

	struct Matrix4x4
	{
		float m[4][4];

		float operator() (unsigned int column, unsigned int row) const
		{
			assert(column < 4 && row < 4);
			return m[column][row];
		}

		float& operator() (unsigned int column, unsigned int row)
		{
			assert(column < 4 && row < 4);
			return m[column][row];
		}
	};
}