# SIMDIntrinsics

[![Build Status](https://travis-ci.org/KristofferC/SIMDIntrinsics.jl.svg?branch=master)](https://travis-ci.org/KristofferC/SIMDIntrinsics.jl)

WIP for experimenting with an API to SIMD and SIMD intrinsics in Julia.

Plan is to have a very low level API that implements the SIMD operations and intrinsics on
Julia's `NTuple{N, VecElement{T}}` and build a higher level API on top of that.

### Random examples:

#### Low level API

Reflects the LLVM operations and intrinsics directly.

```jl
julia> v = LLVM.VE.((1, 2, 3, 4)) # VE short for VecElement
(VecElement{Int64}(1), VecElement{Int64}(2), VecElement{Int64}(3), VecElement{Int64}(4))

julia> LLVM.add(v, v)
(VecElement{Int64}(2), VecElement{Int64}(4), VecElement{Int64}(6), VecElement{Int64}(8))

julia> LLVM.extractelement(v, 2) # zero indexed
3

julia> LLVM.icmp_eq(v, v)
(VecElement{Bool}(true), VecElement{Bool}(true), VecElement{Bool}(true), VecElement{Bool}(true))
```


#### High level API

Uses the LLVM intrinsics and operations to build up a SIMD Vector library:

```julia
julia> using SIMDIntrinsics.SIMD

julia> v = Vec(1.0,2.0,3.0,4.0)
<4 x Float64>[1.0, 2.0, 3.0, 4.0]

julia> v[1]
1.0

julia> Base.setindex(v, 1337.0, 3)
<4 x Float64>[1.0, 2.0, 1337.0, 4.0]

julia> one(Vec{4, Int32})
<4 x Int32>[1, 1, 1, 1]

julia> v + v
<4 x Float64>[2.0, 4.0, 6.0, 8.0]

julia> v + 3.0
<4 x Float64>[4.0, 5.0, 6.0, 7.0]

julia> convert(Vec{4, Int16}, v)
<4 x Int16>[1, 2, 3, 4]

julia> reinterpret(Vec{4, Int64}, v)
<4 x Int64>[4607182418800017408, 4611686018427387904, 4613937818241073152, 4616189618054758400]

julia> shufflevector(v, Val((1,2,1,3)))
<4 x Float64>[2.0, 3.0, 2.0, 4.0]

julia> v2 = Vec(3.0, 4.0, 1.0, 2.0)
<4 x Float64>[3.0, 4.0, 1.0, 2.0]

julia> v2 < v
<4 x Bool>[false, false, true, true]

julia> v3 = Vec(Int8.((0, 2, 1, 3)))

julia> reinterpret(Int32, v3)
50397696

julia> count_ones(v3)
<4 x Int8>[0, 1, 1, 2]

julia> leading_zeros(v3)
<4 x Int8>[8, 6, 7, 6]

julia> a = [1, 2, 3, 4, 5, 6, 7, 8]

julia> vload(Vec{4, Int64}, a, 3)
<4 x Int64>[3, 4, 5, 6]

julia> v4 = Vec(11, 22, 33, 44);

julia> vstore(v4, a, 3);

julia> print(a)
[1, 2, 11, 22, 33, 44, 7, 8]

julia> vstore(v4, a, 10); # boundscheck removed with @inbounds
ERROR: BoundsError: attempt to access 8-element Array{Int64,1} at index [13]
```

### Random thoughts

- Skip the `u` and `i` prefix for LLVM operations and just dispatch on input type?
- Handle alignment
