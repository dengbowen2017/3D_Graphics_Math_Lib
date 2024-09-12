# Outline
This is a basic 3d graphics math library based on SIMD. At first, I tried to use DirectXMath but I found that it isn't convenient enough. As for other math libraries, glm is quite intuitive but it doesn't use SIMD to accelerate the computation while eigen uses SIMD but it is too heavy for computer graphics. Therefore, I decided to improve DirectXMath to implement a new math library.

# How to use
This project is still under development.

# Notice
## Row major vs. Column major
This library uses Column major to store the matrix because the shader files require Column major input for matrix by default. Otherwise, you need to transpose the matrix before you pass it to shader files.

## Reconstruction Plan
After I use this library in my physics engine, I found it is quite ambiguous. For example, there are many matrix3x3 calculations in simulation. If we want to use SIMD to accelerate the calculation, we need to expand the matrix3x3 to matrix4x4 and only set the right-down corner to 1. However, if we treat a expanded matrix4x4 and a normal matrix4x4 as the same one and use the same matrix calculation, there will be a problem. Thinking about matrix subtraction. If we subtract two expanded matrix4x4 as same as normal matrix4x4, the right-down corner will be 0, but it should be 1 since we are actually doing matrix3x3 subtraction. 

Therefore, to avoid this kind of ambiguity, I decide to reconstruct the math library to make it more consistent with mathematics. The new library will only provide Vector3, Vector4, Mat3x3, Mat4x4 with intrinsics or without intrinsics (you can switch it by using macro definition).

# Blog
[Some thoughts about C++ and Intrinsics](./doc/blog.md)