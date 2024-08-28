#include "GMath.h"

#include <iostream>
#include <chrono>

// benchmark
#include <DirectXMath.h>

int main()
{
	auto start = std::chrono::high_resolution_clock::now();

	GMath::MVector a(1.0f);
	GMath::MVector b(1.0f, 2.0f, 3.0f, 4.0f);

	GMath::MVector c = a + b;

	GMath::Vector4 res;
	GMath::StoreVector4(&res, c);

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << duration.count() << " microseconds" << std::endl;

	return 0;
}