#include "GMathLib.h"

#include <iostream>
#include <chrono>

// benchmark
#include <DirectXMath.h>

int main()
{
	GMath::Vector4 x(1.0f, 2.0f, 3.0f, 4.0f);
	GMath::Vector4 y(2.0f, 2.0f, 3.0f, 4.0f);
	GMath::Vector4 z(0.0f);

	//__m128 a = _mm_loadu_ps(&x.x);
	//__m128 b = _mm_loadu_ps(&y.x);
	//__m128 c = _mm_add_ps(a, b);
	//_mm_storeu_ps(&z.x, c);

	z = y;

	std::cout << z.x << std::endl;
	std::cout << z.y << std::endl;
	std::cout << z.z << std::endl;
	std::cout << z.w << std::endl;

	return 0;
}
