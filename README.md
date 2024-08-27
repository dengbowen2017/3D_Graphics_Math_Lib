# Outline
This is a basic 3d graphics math library based on SIMD. At first, I tried to use DirectXMath but I found that it isn't convenient enough. As for other math libraries, glm is quite intuitive but it doesn't use SIMD to accelerate the computation while eigen uses SIMD but it is too heavy for computer graphics. Therefore, I decided to improve DirectXMath to implement a new math library.

# How to use
This project is still under development.

# Notice
## Row major vs. Column major
This library uses Column major to store the matrix because the shader files require Column major input for matrix by default. Otherwise, you need to transpose the matrix before you pass it to shader files.

# Blog
[Some thoughts about C++ and Intrinsics](./doc/blog.md)