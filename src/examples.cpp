#include "GMathLib.h"

#include <iostream>
#include <chrono>

// benchmark
#include <DirectXMath.h>

int main()
{
	auto start = std::chrono::high_resolution_clock::now();


	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << duration.count() << " microseconds" << std::endl;

	return 0;
}