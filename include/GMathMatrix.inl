#pragma once

// -----------------------------
// Matrix Operations
// -----------------------------

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

	MMatrix M;
	M.r[0] = _mm_blend_ps(D0.v, R0.v, 0x07);
	M.r[1] = _mm_blend_ps(D1.v, R1.v, 0x07);
	M.r[2] = _mm_sub_ps(g_VecAllZero, _mm_blend_ps(D2.v, R2.v, 0x07));
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

// --------------------------
// Matrix Overloads
// --------------------------

inline MMatrix MMatrix::operator-() const noexcept
{
	MMatrix M;
	M.r[0] = _mm_sub_ps(g_VecAllZero, this->r[0]);
	M.r[1] = _mm_sub_ps(g_VecAllZero, this->r[1]);
	M.r[2] = _mm_sub_ps(g_VecAllZero, this->r[2]);
	M.r[3] = _mm_sub_ps(g_VecAllZero, this->r[3]);
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
	return MatrixMulVector(*this, V);
}