# Imitate the API in SIMD.jl

module SIMD

import ..SIMDIntrinsics: LLVM, VE, LVec, ScalarTypes, IntegerTypes, IntTypes, UIntTypes, FloatingTypes
import .LLVM: shufflevector

using Base: @propagate_inbounds

export Vec, vload, vstore, vgather, vscatter, vload, shufflevector

struct Vec{N, T <: ScalarTypes}
    data::LVec{N, T}
end
Vec(v::NTuple{N, T}) where {N, T <: ScalarTypes} = Vec(VE.(v))
Vec(v::Vararg{T, N}) where {N, T <: ScalarTypes} = Vec(v)
Vec(v::Vec) = v
Vec{N, T}(v::Vec{N, T}) where {N, T<:ScalarTypes} = v
Vec{N, T1}(v::Vec{N, T2}) where {N, T1<:ScalarTypes, T2<:ScalarTypes} = convert(Vec{N, T1}, v)
Vec{N, T}(v::T) where {N, T<:ScalarTypes} = fill(v, Vec{N, T})

# Should promotion be supported?
#=
@inline function Base.promote_rule(::Type{Vec{N, T1}}, ::Type{Vec{N, T2}}) where {T1, T2, N}
    Vec{N, promote_type(T1, T2)}
end

@inline function Base.promote(v1::Vec{N, T1}, v2::Vec{N, T2}) where {T1, T2, N}
    return convert(promote_type(Vec{N, T1}, Vec{N, T2}), v1),
           convert(promote_type(Vec{N, T1}, Vec{N, T2}), v2)
end
=#

# noop convert
@inline Base.convert(::Type{Vec{N,T}}, v::Vec{N,T}) where {N,T} = v

#
Base.Tuple(v::Vec) = map(i -> i.value, v.data)
Base.NTuple{N, T}(v::Vec{N}) where {T, N} = map(i -> convert(T, i.value), v.data)

# No checks for underflow or overflow!
@inline function Base.convert(::Type{Vec{N, T1}}, v::Vec{N, T2}) where {T1, T2, N}
    if T1 <: IntegerTypes
        if T2 <: IntegerTypes
            if sizeof(T1) < sizeof(T2)
                return Vec(LLVM.trunc(LLVM.LVec{N, T1}, v.data))
            else
                return Vec(LLVM.sext(LLVM.LVec{N, T1}, v.data))
            end
        elseif T2 <: FloatingTypes
            if T1 <: UIntTypes
                return Vec(LLVM.fptoui(LLVM.LVec{N, T1}, v.data))
            elseif T1 <: IntTypes
                return Vec(LLVM.fptosi(LLVM.LVec{N, T1}, v.data))
            end
        end
    end
    if T1 <: FloatingTypes
        if T2 <: UIntTypes
            return Vec(LLVM.uitofp(LLVM.LVec{N, T1}, v.data))
        elseif T2 <: IntTypes
            return Vec(LLVM.sitofp(LLVM.LVec{N, T1}, v.data))
        elseif T2 <: FloatingTypes
            if sizeof(T1) < sizeof(T2)
                return Vec(LLVM.fptrunc(LLVM.LVec{N, T1}, v.data))
            else
                return Vec(LLVM.fpext(LLVM.LVec{N, T1}, v.data))
            end
        end
    end
    error("unreachable")
end


Base.eltype(::Type{Vec{N,T}}) where {N,T} = T
Base.ndims( ::Type{Vec{N,T}}) where {N,T} = 1
Base.length(::Type{Vec{N,T}}) where {N,T} = N
Base.size(  ::Type{Vec{N,T}}) where {N,T} = (N,)
# TODO: This doesn't follow Base, e.g. `size([], 3) == 1`
Base.size(::Type{Vec{N,T}}, n::Integer) where {N,T} = (N,)[n]

Base.eltype(V::Vec) = eltype(typeof(V))
Base.ndims( V::Vec) = ndims(typeof(V))
Base.length(V::Vec) = length(typeof(V))
Base.size(  V::Vec) = size(typeof(V))
Base.size(  V::Vec, n::Integer) = size(typeof(V), n)

function Base.show(io::IO, v::Vec{N,T}) where {N,T}
    print(io, "<$N x $T>[")
    join(io, [x.value for x in v.data], ", ")
    print(io, "]")
end

@inline Base.checkbounds(v::Vec, i::IntegerTypes) =
(i < 1 || i > length(v.data)) && Base.throw_boundserror(v, i)

function Base.getindex(v::Vec, i::IntegerTypes)
    @boundscheck checkbounds(v, i)
    return LLVM.extractelement(v.data, i-1)
end

@propagate_inbounds Base.getindex(v::Vec, ::Val{i}) where {i} = v[i]
@propagate_inbounds Base.getindex(v::Vec, ::Type{Val{i}}) where {i} = v[i]

@inline function Base.setindex(v::Vec{N,T}, x, i::IntegerTypes) where {N,T}
    @boundscheck checkbounds(v, i)
    Vec(LLVM.insertelement(v.data, convert(T, x), i-1))
end

function Base.setindex(v::Vec{N, T}, x, ::Val{i}) where {N,T,i}
    @boundscheck checkbounds(v, i)
    Vec(LLVM.insertelement(v.data, convert(T, x), Val(i)))
end
@propagate_inbounds function Base.setindex(v::Vec, x, ::Type{Val{i}}) where {i}
    Base.setindex(v, x, Val(i))
end

Base.zero(::Type{Vec{N,T}}) where {N, T} = fill(zero(T), Vec{N, T})
Base.zero(::Vec{N,T}) where {N, T} = zero(Vec{N, T})
Base.one(::Type{Vec{N,T}}) where {N, T} = fill(one(T), Vec{N, T})
Base.one(::Vec{N,T}) where {N, T} = one(Vec{N, T})

Base.reinterpret(::Type{Vec{N1, T1}}, v::Vec) where {T1, N1} = Vec(LLVM.bitcast(LLVM.LVec{N1, T1}, v.data))
Base.reinterpret(::Type{T}, v::Vec) where {T} = LLVM.bitcast(T, v.data)

const BINARY_OPS = [
    (:+, IntegerTypes, LLVM.add)
    (:-, IntegerTypes, LLVM.sub)
    (:*, IntegerTypes, LLVM.mul)
    (:div, UIntTypes, LLVM.udiv)
    (:div, IntTypes, LLVM.sdiv)
    (:rem, UIntTypes, LLVM.urem)
    (:rem, IntTypes, LLVM.srem)

    (:+, FloatingTypes, LLVM.fadd)
    (:-, FloatingTypes, LLVM.fsub)
    (:*, FloatingTypes, LLVM.fmul)
    (:^, FloatingTypes, LLVM.pow)
    (:/, FloatingTypes, LLVM.fdiv)
    (:rem, FloatingTypes, LLVM.frem)
    (:min, FloatingTypes, LLVM.minnum)
    (:max, FloatingTypes, LLVM.maxnum)

    (:~, IntegerTypes, LLVM.xor)
    (:&, IntegerTypes, LLVM.and)
    (:|, IntegerTypes, LLVM.or)
    (:⊻, IntegerTypes, LLVM.xor)

    (:<<, IntegerTypes, LLVM.shl)
    (:>>>, IntegerTypes, LLVM.lshr)
    (:>>, UIntTypes, LLVM.lshr)
    (:>>, IntTypes, LLVM.ashr)

    (:(==), IntegerTypes, LLVM.icmp_eq)
    (:(!=), IntegerTypes, LLVM.icmp_ne)
    (:(>), IntTypes, LLVM.icmp_sgt)
    (:(>=), IntTypes, LLVM.icmp_sge)
    (:(<), IntTypes, LLVM.icmp_slt)
    (:(<=), IntTypes, LLVM.icmp_sle)
    (:(>), UIntTypes, LLVM.icmp_ugt)
    (:(>=), UIntTypes, LLVM.icmp_uge)
    (:(<), UIntTypes, LLVM.icmp_ult)
    (:(<=), UIntTypes, LLVM.icmp_ule)

    (:(==), FloatingTypes, LLVM.fcmp_oeq)
    (:(!=), FloatingTypes, LLVM.fcmp_une)
    (:(>), FloatingTypes, LLVM.fcmp_ogt)
    (:(>=), FloatingTypes, LLVM.fcmp_oge)
    (:(<), FloatingTypes, LLVM.fcmp_olt)
    (:(<=), FloatingTypes, LLVM.fcmp_ole)
]

for (op, constraint, llvmop) in BINARY_OPS
    @eval @inline function (Base.$op)(x::Vec{N, T}, y::Vec{N, T}) where {N, T <: $constraint}
        Vec($(llvmop)(x.data, y.data))
    end
    @eval @inline function (Base.$op)(x::T, y::Vec{N, T}) where {N, T <: $constraint}
        Vec($(llvmop)(fill(x, Vec{N, T}).data, y.data))
    end
    @eval @inline function (Base.$op)(x::Vec{N, T}, y::T) where {N, T <: $constraint}
        Vec($(llvmop)(x.data, fill(y, Vec{N, T}).data))
    end
end

Base.fill(v::T, ::Type{Vec{N, T}}) where {N, T} = Vec(LLVM.constantvector(v, LLVM.LVec{N, T}))

@inline vifelse(v::Bool, v1::Vec{N, T}, v2::Vec{N, T}) where {N, T} = ifelse(v, v1, v2)
@inline vifelse(v::Vec{N, Bool}, v1::Vec{N, T}, v2::Vec{N, T}) where {N, T} =
    Vec(LLVM.select(v.data, v1.data, v2.data))

@inline Base.max(v1::Vec{N,T}, v2::Vec{N,T}) where {N,T<:IntegerTypes} =
    Vec(vifelse(v1 >= v2, v1, v2))
@inline Base.min(v1::Vec{N,T}, v2::Vec{N,T}) where {N,T<:IntegerTypes} =
    Vec(vifelse(v1 >= v2, v2, v1))

# Pow
@inline Base.:^(x::Vec{N,T}, y::IntegerTypes) where {N,T<:FloatingTypes} =
    Vec(LLVM.powi(x.data, y))

@inline Base.literal_pow(::typeof(^), x::Vec{Any, <:FloatingTypes}, ::Val{0}) = one(typeof(x))
@inline Base.literal_pow(::typeof(^), x::Vec{Any, <:FloatingTypes}, ::Val{1}) = x
@inline Base.literal_pow(::typeof(^), x::Vec{Any, <:FloatingTypes}, ::Val{2}) = x*x
@inline Base.literal_pow(::typeof(^), x::Vec{Any, <:FloatingTypes}, ::Val{3}) = x*x*x


#########
# Unary #
#########

const UNARY_OPS = [
    (:sqrt, FloatingTypes, LLVM.sqrt),
    (:sin, FloatingTypes, LLVM.sin),
    (:cos, FloatingTypes, LLVM.cos),
    (:exp, FloatingTypes, LLVM.exp),
    (:exp2, FloatingTypes, LLVM.exp2),
    (:log, FloatingTypes, LLVM.log),
    (:log10, FloatingTypes, LLVM.log10),
    (:log2, FloatingTypes, LLVM.log2),
    (:abs, FloatingTypes, LLVM.fabs),
    (:floor, FloatingTypes, LLVM.floor),
    (:ceil, FloatingTypes, LLVM.ceil),
    # (:rint, FloatingTypes, LLVM),
    # (:nearbyint, FloatingTypes, LLVM),
    (:round, FloatingTypes, LLVM.round),

    #:bitreverse,
    (:bswap, IntegerTypes, LLVM.bswap),
    (:count_ones, IntegerTypes, LLVM.ctpop),
    (:leading_zeros, IntegerTypes, LLVM.ctlz),
    (:trailing_zeros, IntegerTypes, LLVM.cttz),
    (:~, IntegerTypes, LLVM.or)
    #:fshl,
    #:fshr,
]

for (op, constraint, llvmop) in UNARY_OPS
    @eval @inline (Base.$op)(x::Vec{<:Any, <:$constraint}) =
        Vec($(llvmop)(x.data))
end

Base.:+(v::Vec) = v
Base.:-(v::Vec) = zero(v) - v
Base.abs(v::Vec{N, T}) where {N, T} = Vec(vifelse(v < zero(T), -v, v))
Base.:!(v1::Vec{N,Bool}) where {N} = ~v1
Base.inv(v::Vec{N, T}) where {N, T<:FloatingTypes} = one(T) / v


##############
# Reductions #
##############
const HORZ_REDUCTION_OPS = [
    (&, IntegerTypes, LLVM.reduce_and)
    (|, IntegerTypes, LLVM.reduce_and)
    (max, IntTypes, LLVM.reduce_smax)
    (max, UIntTypes,LLVM.reduce_umax)
    (max, FloatingTypes, LLVM.reduce_fmax)
    (min, IntTypes, LLVM.reduce_smin)
    (min, UIntTypes, LLVM.reduce_umin)
    (min, FloatingTypes, LLVM.reduce_fmin)
    (+, IntegerTypes, LLVM.reduce_add)
    (*, IntegerTypes, LLVM.reduce_mul)
    (+, FloatingTypes, LLVM.reduce_fadd)
    (*, FloatingTypes, LLVM.reduce_fmul)
]

for (op, constraint, llvmop) in HORZ_REDUCTION_OPS
    @eval @inline Base.reduce(::typeof($op), x::Vec{<:Any, <:$constraint}) =
        $(llvmop)(x.data)
end
Base.reduce(F::Any, v::Vec) = error("reduction not defined for SIMD.Vec on $F")

@inline Base.all(v::Vec{<:Any,Bool}) = reduce(&, v)
@inline Base.any(v::Vec{<:Any,Bool}) = reduce(|, v)
@inline Base.maximum(v::Vec) = reduce(max, v)
@inline Base.minimum(v::Vec) = reduce(min, v)
@inline Base.prod(v::Vec) = reduce(*, v)
@inline Base.sum(v::Vec) = reduce(+, v)

#################################################

# Various bit counts defined in terms of others
@inline Base.leading_ones(x::Vec{<:Any, <:IntegerTypes})  = leading_zeros(~(x))
@inline Base.trailing_ones(x::Vec{<:Any, <:IntegerTypes}) = trailing_zeros(~(x))
@inline Base.count_zeros(x::Vec{<:Any, <:IntegerTypes}) = count_zeros(~(x))

@inline Base.isnan(v::Vec{<:Any, <:FloatingTypes}) = v != v
@inline Base.isfinite(v::Vec{<:Any, <:FloatingTypes}) = v - v == zero(v)
@inline Base.isinf(v::Vec{<:Any, <:FloatingTypes}) = !isnan(v) & !isfinite(v)
@inline Base.signbit(x::Vec{<:Any})  = x < zero(x)
@inline Base.sign(v1::Vec{N,T}) where {N,T} =
    vifelse(v1 == zero(Vec{N,T}), zero(Vec{N,T}),
            vifelse(v1 < zero(Vec{N,T}), -one(Vec{N,T}), one(Vec{N,T})))

@inline Base.isnan(v::Vec{N, <:IntegerTypes}) where {N} = zero(Vec{N,Bool})
@inline Base.isfinite(v::Vec{N, <:IntegerTypes}) where {N} = one(Vec{N, Bool})
@inline Base.isinf(v::Vec{N, <:IntegerTypes}) where {N} = zero(Vec{N, Bool})

@inline Base.copysign(v1::Vec{N,T}, v2::Vec{N,T}) where {N,T} =
    vifelse(signbit(v2), -abs(v1), abs(v1))
@inline Base.flipsign(v1::Vec{N,T}, v2::Vec{N,T}) where {N,T} =
    vifelse(signbit(v2), -v1, v1)

# Muladd
Base.muladd(a::Vec{N, T}, b::Vec{N, T}, c::Vec{N, T}) where {N,T} = Vec(LLVM.fmuladd(a.data, b.data, c.data))
Base.fma(a::Vec{N, T}, b::Vec{N, T}, c::Vec{N, T}) where {N,T} = Vec(LLVM.fma(a.data, b.data, c.data))

@inline vload(::Type{Vec{N, T}}, ptr::Ptr{T}) where {N, T} = Vec(LLVM.load(LLVM.LVec{N, T}, ptr))
@inline function vload(::Type{Vec{N, T}}, a::Array{T}, i::Integer) where {N, T}
    @boundscheck checkbounds(a, i + N - 1)
    GC.@preserve a begin
        return vload(Vec{N, T}, pointer(a, i))
    end
end

@inline vstore(x::Vec{N, T}, ptr::Ptr{T}) where {N, T} = LLVM.store(x.data, ptr)
@inline function vstore(x::Vec{N, T}, a::Array{T}, i::Integer) where {N, T}
    @boundscheck checkbounds(a, i + N - 1)
    GC.@preserve a begin
        vstore(x, pointer(a, i))
    end
    return a
end

@inline function LLVM.shufflevector(x::Vec{N, T}, ::Val{I}) where {N, T, I}
    Vec(LLVM.shufflevector(x.data, Val(I)))
end
@inline function LLVM.shufflevector(x::Vec{N, T}, y::Vec{N, T}, ::Val{I}) where {N, T, I}
    Vec(LLVM.shufflevector(x.data, y.data, Val(I)))
end

@inline function vgather(a::Array{T}, group_lane::Vec{N, Int}) where {N, T}
    @boundscheck for i in 1:N
        checkbounds(a, @inbounds group_lane[i])
    end
    p = fill(Int(pointer(a)), Vec{N, Int})
    ptrs = p + (group_lane - 1) * sizeof(T)
    GC.@preserve a begin
        return Vec(LLVM.maskedgather(LLVM.LVec{N, T}, ptrs.data))
    end
end

@inline function vscatter(a::Array{T}, x::Vec{N, T}, group_lane::Vec{N, Int}) where {N, T}
    @boundscheck for i in 1:N
        checkbounds(a, @inbounds group_lane[i])
    end
    p = fill(Int(pointer(a)), Vec{N, Int})
    ptrs = p + (group_lane - 1) * sizeof(T)
    GC.@preserve a begin
        LLVM.maskedscatter(x.data, ptrs.data)
    end
    return
end


end
