#pragma once

// ----------------------------
// Vector Operations
// ----------------------------

inline float __vectorcall VectorGetX(const MVector V) noexcept
{
	return _mm_cvtss_f32(V.v);
}

inline MVector __vectorcall VectorNegate(const MVector V) noexcept
{
	MVector vRes;
	vRes.v = _mm_sub_ps(g_VecAllZero, V.v);
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
	__m128 vZeroMask = _mm_cmpneq_ps(g_VecAllZero, vResult);
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

inline MVector __vectorcall MatrixMulVector(const MMatrix M, const MVector V) noexcept
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

inline MMatrix __vectorcall VectorCrossToMatrix(const MVector V) noexcept
{
	MMatrix mRes;

	__m128 vValue = _mm_and_ps(V.v, g_VecMaskXYZ);
	__m128 vNegate = _mm_sub_ps(g_VecAllZero, vValue);

	mRes.r[0] = _mm_shuffle_ps(vValue, vNegate, _MM_SHUFFLE(3, 1, 2, 3));
	mRes.r[1] = _mm_shuffle_ps(vNegate, vValue, _MM_SHUFFLE(3, 0, 3, 2));

	vValue = _mm_shuffle_ps(vValue, vValue, _MM_SHUFFLE(3, 3, 3, 1));
	vNegate = _mm_shuffle_ps(vNegate, vNegate, _MM_SHUFFLE(3, 3, 0, 3));
	mRes.r[2] = _mm_add_ps(vValue, vNegate);
	mRes.r[3] = g_MatIdentityR3;

	return mRes;
}

// -----------------------
// Vector Overloads
// -----------------------

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