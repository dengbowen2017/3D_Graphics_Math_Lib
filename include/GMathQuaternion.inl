#pragma once

// -----------------------------
// Quaternion Operations
// -----------------------------

inline MQuaternion __vectorcall QuaternionMul(const MQuaternion Q1, const MQuaternion Q2) noexcept
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

inline MMatrix __vectorcall QuaternionToMatrix(const MQuaternion Q) noexcept
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

inline MQuaternion __vectorcall QuaternionNormalize(const MQuaternion Q) noexcept
{
	MQuaternion qRes;
	__m128 qLengthSq = _mm_dp_ps(Q.q, Q.q, 0xff);
	__m128 qResult = _mm_sqrt_ps(qLengthSq);
	__m128 qZeroMask = _mm_cmpneq_ps(g_VecAllZero, qResult);
	qResult = _mm_div_ps(Q.q, qResult);
	qRes.q = _mm_and_ps(qResult, qZeroMask);
	return qRes;
}

// -------------------------------
// Quaternion Overloads
// -------------------------------

inline MQuaternion MQuaternion::operator-() const noexcept
{
	MQuaternion qRes;
	qRes.q = _mm_sub_ps(g_VecAllZero, this->q);
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