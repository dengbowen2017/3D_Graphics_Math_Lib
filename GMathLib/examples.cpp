#include "GMathLib.h"

#include <iostream>
#include <chrono>

// benchmark
#include <DirectXMath.h>

int main()
{
	//GMath::Vector3A vec1(1);
	//__m128 b = *(__m128*)&vec1;

	GMath::Vector3A v1(3.0f);
	GMath::Vector4 v2(2.0f);
	GMath::Vector4 v;

	//auto start = std::chrono::high_resolution_clock::now();

	__m128 b = GMath::LoadVector3A(&v1);
	__m128 c = GMath::LoadVector4(&v2);
	__m128 res = _mm_mul_ps(b, c);
	GMath::StoreVector4(&v, res);

	//auto end = std::chrono::high_resolution_clock::now();
	//auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	//std::cout << duration.count() << " microseconds" << std::endl;
	return 0;
}
