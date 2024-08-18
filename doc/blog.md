## C++
### constexpr value
A *constexpr integral value* can be used wherever a const integer is required, such as in template arguments and array declarations. **And when a value is computed at compile time instead of run time, it helps your program run faster and use less memory.**

### constexpr function
A *constexpr function* is one whose return value is computable at compile time when consuming code requires it. 

A *inline function* is used to suggest to the compiler that it should attempt to expand a function in place, rather than calling it through the usual function call mechanism. This can lead to performance improvements, especially for very small, frequently called functions, by reducing the overhead associated with function calls.

So, a *constexpr function* is computed at compile time while a *inline function* just avoids creating/deleting stack frame at run time, which means a *constexpr function* will be faster than a *inline function*.

### __vectorcall
Usually, the procedure of calling a function will include many steps
- set up stack frame
- push parameters into stack frame from right to left
- push return address into stack frame
- push local variables into stack frame
- execute function
- store results in a register
- clean stack frame
- return to the caller 

The __vectorcall calling convention specifies that arguments to functions are to be passed in registers, when possible. __vectorcall calling convention is only supported in native code on x86 and x64 processors that include Streaming SIMD Extensions 2 (SSE2) and above. Use __vectorcall to speed functions that pass several floating-point or SIMD vector arguments and perform operations that take advantage of the arguments loaded in registers. 

In short, __vectorcall functions try to pass values directly to registers without using stack frame, so __vectorcall functions will be inline functions and also will be faster than __stdcall functions in which registers have to read values from stack frame. But if the parameters of a __vectorcall function are too much, some values will be passed by stack frame.

**Passing by Value vs. Passing by Reference**

- Passing by Value:
When you pass an int by value, the compiler will typically place the actual value of the int directly into a CPU register (if registers are available according to the calling convention).
The function can then operate directly on this value in the register, with no additional indirection or dereferencing required.
- Passing by Reference:
When you pass an int by reference, what actually gets passed to the function is a reference (or pointer) to the original int.
The reference itself might be stored in a register, but the function must dereference this reference (i.e., access the memory location it points to) to get the actual value of the int.
This adds a layer of indirection, as the CPU has to load the value from the memory address stored in the register before performing any operations on it.

So, For small types, we pass them by value. For big types, we pass them by reference. Passing an int by value allows the function to use the value directly in the register, avoiding the indirection that would come with passing by reference.

You can pass three kinds of arguments by register in __vectorcall functions: integer type values, vector type values, and homogeneous vector aggregate (HVA) values. Integer types include pointer, reference, and struct or union types of 4 bytes (8 bytes on x64) or less. A vector type is either a floating-point type—for example, a float or double—or an SIMD vector type—for example, __m128 or __m256. An HVA type is a composite type of up to four data members that have identical vector types.

### __declspec(selectany)
Usually, we have to initialize a extern const value in a cpp file not a header file but with this declaration we can do it in a header file.

## Intrinsics

### __m128

SSE only supports __m128 while AVX supports both __m128 and __m256 but this lib is a 3D graphics math lib so we will only use __m128 which is supported by almost all the platforms. It's because in graphics math we will only deal with vector with 3 or 4 floats and matrix with 3x3 floats or 4x4 floats. 

**The definition of __m128**
```
typedef union __declspec(intrin_type) __declspec(align(16)) __m128 {
     float               m128_f32[4];
     unsigned __int64    m128_u64[2];
     __int8              m128_i8[16];
     __int16             m128_i16[8];
     __int32             m128_i32[4];
     __int64             m128_i64[2];
     unsigned __int8     m128_u8[16];
     unsigned __int16    m128_u16[8];
     unsigned __int32    m128_u32[4];
 } __m128;
```
So, __m128 is just a 128 bits (16 bytes) union, or you can say it is just a block of memory, in RAM aligned on a 16-byte boundary, which means you can cast a pointer that points to an aligned 16-byte memory to __m128* and use the __m128* without any problem or vice versa. In this way, you can avoid to use load/store intrinsic instructions. 

If you use VS, you can open Memory Debug Window and check the address and content of the __m128. If a structure or data type is aligned on a 16-byte boundary, it means the memory address where this data begins is a multiple of 16, which means the lowest number of the address will always be 0. You can use **__declspec(align(16))** before the definition of your struct to define a struct aligned on a 16-byte boundary.

**How to initialize __m128**

There are two ways. One is using _mm_set_ps. The other one is using initializer list. I test the overhead of these two ways and it turns out to be that _mm_set_ps is way more faster than initializer list. So, I use _mm_set_ps to initialize __m128.

### Load Instruction 
slow (high latency about 6 CPU cycles) but high throughput (about 3 Instruction at 1 CPU cycle)

### Unpack vs. Move
assume that A: a4 a3 a2 a1, B: b4 b3 b2 b1

Unpackhi: b4 a4 b3 a3, Unpacklo: b2 a2 b1 a1

movehl: a4 a3 b4 b3, movelh: b2 b1 a2 a1