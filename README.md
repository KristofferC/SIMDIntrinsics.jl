# SIMDIntrinsics

[![Build Status](https://travis-ci.org/KristofferC/SIMDIntrinsics.jl.svg?branch=master)](https://travis-ci.org/KristofferC/SIMDIntrinsics.jl)

WIP for experimenting with an API to SIMD and SIMD intrinsics in Julia.

Plan is to have a very low level API that implements the SIMD operations and intrinsics on
Julia's `NTuple{N, VecElement{T}}` and build a higher level API on top of that.

Thoughts:

- Skip the `u` and `i` prefix for LLVM operations and just dispatch on input type?
