module SIMDIntrinsics

# TODO SIMD intrinsics

const VE = Base.VecElement
const Vec{N, T} = NTuple{N, Base.VecElement{T}}

const BoolTypes = Union{Bool}
const IntTypes = Union{Int8, Int16, Int32, Int64, Int128}
const UIntTypes = Union{UInt8, UInt16, UInt32, UInt64, UInt128}
const IntegerTypes = Union{BoolTypes, IntTypes, UIntTypes}
const FloatingTypes = Union{Float32, Float64} # Float16 support is non-native in Julia gets passed as an i16

include("LLVM.jl")
include("intrin/immintrin.jl")

end