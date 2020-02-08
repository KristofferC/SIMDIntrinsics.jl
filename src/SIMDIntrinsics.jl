module SIMDIntrinsics

const VE = Base.VecElement
const LVec{N, T} = NTuple{N, Base.VecElement{T}}

# From SIMD.jl
const BoolTypes = Union{Bool}
const IntTypes = Union{Int8, Int16, Int32, Int64, Int128}
const UIntTypes = Union{UInt8, UInt16, UInt32, UInt64, UInt128}
const IntegerTypes = Union{BoolTypes, IntTypes, UIntTypes}
const FloatingTypes = Union{Float32, Float64} # Float16 support is non-native in Julia gets passed as an i16
const ScalarTypes = Union{IntegerTypes, FloatingTypes}

include("LLVM.jl")
include("SIMD.jl")
include("intrin/immintrin.jl")

using .SIMD

export Vec, vload, vstore, vgather, vscatter, vload, shufflevector


end
