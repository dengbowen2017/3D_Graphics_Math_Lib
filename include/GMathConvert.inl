#pragma once

// -------------------------
// Load Operations
// -------------------------

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

// ---------------------------
// Store Operations
// ---------------------------

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