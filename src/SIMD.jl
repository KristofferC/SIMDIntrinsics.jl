# Imitate the API in SIMD.jl

module SIMD

import ..SIMDIntrinsics: LLVM, VE, LVec, ScalarTypes, IntegerTypes, IntTypes, UIntTypes, FloatingTypes
using Base: @propagate_inbounds

    export Vec, vload, vstore, vgather, vscatter, vload, shufflevector, vifelse, valloc

struct Vec{N, T <: ScalarTypes}
    data::LVec{N, T}
end

# Constructors
@inline Vec(v::NTuple{N, T}) where {N, T <: ScalarTypes} = Vec(VE.(v))
@inline Vec(v::Vararg{T, N}) where {N, T <: ScalarTypes} = Vec(v)
@inline Vec(v::Vec) = v

# No throwing versions of convert
@inline _unsafe_convert(::Type{T}, v) where {T <: IntegerTypes} = v % T
@inline _unsafe_convert(::Type{T}, v) where {T <: FloatingTypes} = convert(T, v)
@inline constantvector(v::T1, ::Type{Vec{N, T2}}) where {N, T1, T2} =
    Vec(LLVM.constantvector(_unsafe_convert(T2, v), LLVM.LVec{N, T2}))

@inline Vec{N, T}(v::Vec{N, T}) where {N, T<:IntegerTypes} = v
@inline Vec{N, T}(v::Vec{N, T}) where {N, T<:FloatingTypes} = v
@inline Vec{N, T1}(v::T2) where {N, T1<:ScalarTypes, T2<:ScalarTypes} = constantvector(v, Vec{N, T1})
@inline Vec{N, T1}(v::Vec{N, T2}) where {N, T1<:IntegerTypes, T2<:IntegerTypes} = convert(Vec{N, T1}, v)

include("vectorops.jl")

@inline Base.convert(::Type{Vec{N,T}}, v::Vec{N,T}) where {N,T} = v
@inline function Base.convert(::Type{Vec{N, T1}}, v::Vec{N, T2}) where {T1, T2, N}
    if T1 <: IntegerTypes
        if T2 <: IntegerTypes
            if sizeof(T1) < sizeof(T2)
                return Vec(LLVM.trunc(LLVM.LVec{N, T1}, v.data))
            elseif sizeof(T1) == sizeof(T2)
                return Vec(LLVM.bitcast(LLVM.LVec{N, T1}, v.data))
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
    _unreachable()
end
@noinline _unreachable() = error("unreachable")

Base.Tuple(v::Vec) = map(i -> i.value, v.data)
Base.NTuple{N, T}(v::Vec{N}) where {T, N} = map(i -> convert(T, i.value), v.data)

Base.eltype(::Type{Vec{N,T}}) where {N,T} = T
Base.ndims( ::Type{Vec{N,T}}) where {N,T} = 1
Base.length(::Type{Vec{N,T}}) where {N,T} = N
Base.size(  ::Type{Vec{N,T}}) where {N,T} = (N,)
Base.size(  ::Type{Vec{N,T}}, n::Integer) where {N,T} = n > N ? 1 : (N,)[n]

Base.eltype(V::Vec) = eltype(typeof(V))
Base.ndims(V::Vec) = ndims(typeof(V))
Base.length(V::Vec) = length(typeof(V))
Base.size(V::Vec) = size(typeof(V))
Base.size(V::Vec, n::Integer) = size(typeof(V), n)

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

@inline function Base.setindex(v::Vec{N,T}, x, i::IntegerTypes) where {N,T}
    @boundscheck checkbounds(v, i)
    Vec(LLVM.insertelement(v.data, _unsafe_convert(T, x), i-1))
end

Base.zero(::Type{Vec{N,T}}) where {N, T} = Vec{N,T}(zero(T))
Base.zero(::Vec{N,T}) where {N, T} = zero(Vec{N, T})
Base.one(::Type{Vec{N,T}}) where {N, T} = Vec{N, T}(one(T))
Base.one(::Vec{N,T}) where {N, T} = one(Vec{N, T})

Base.reinterpret(::Type{Vec{N, T}}, v::Vec) where {T, N} = Vec(LLVM.bitcast(LLVM.LVec{N, T}, v.data))
Base.reinterpret(::Type{T}, v::Vec) where {T} = LLVM.bitcast(T, v.data)


###################
# Unary operators #
###################

const UNARY_OPS = [
    (:sqrt           , FloatingTypes , LLVM.sqrt)       ,
    (:sin            , FloatingTypes , LLVM.sin)        ,
    (:trunc          , FloatingTypes , LLVM.trunc)      ,
    (:cos            , FloatingTypes , LLVM.cos)        ,
    (:exp            , FloatingTypes , LLVM.exp)        ,
    (:exp2           , FloatingTypes , LLVM.exp2)       ,
    (:log            , FloatingTypes , LLVM.log)        ,
    (:log10          , FloatingTypes , LLVM.log10)      ,
    (:log2           , FloatingTypes , LLVM.log2)       ,
    (:abs            , FloatingTypes , LLVM.fabs)       ,
    (:floor          , FloatingTypes , LLVM.floor)      ,
    (:ceil           , FloatingTypes , LLVM.ceil)       ,
    # (:rint         , FloatingTypes , LLVM)            ,
    # (:nearbyint    , FloatingTypes , LLVM)            ,
    (:round          , FloatingTypes , LLVM.round)      ,

    # (:bitreverse   , IntegerTypes  , LLVM.bitreverse) ,
    (:bswap          , IntegerTypes  , LLVM.bswap)      ,
    (:count_ones     , IntegerTypes  , LLVM.ctpop)      ,
    (:leading_zeros  , IntegerTypes  , LLVM.ctlz)       ,
    (:trailing_zeros , IntegerTypes  , LLVM.cttz)       ,
    (:~              , IntegerTypes  , LLVM.or)
]

for (op, constraint, llvmop) in UNARY_OPS
    @eval @inline (Base.$op)(x::Vec{<:Any, <:$constraint}) =
        Vec($(llvmop)(x.data))
end

Base.:+(v::Vec) = v
Base.:-(v::Vec{<:Any, <:IntegerTypes}) = zero(v) - v
Base.:-(v::Vec{<:Any, <:FloatingTypes}) = Vec(LLVM.fneg(v.data))
Base.abs(v::Vec{N, T}) where {N, T} = Vec(vifelse(v < zero(T), -v, v))
Base.:!(v1::Vec{N,Bool}) where {N} = ~v1
Base.inv(v::Vec{N, T}) where {N, T<:FloatingTypes} = one(T) / v

_unsigned(::Type{Float32}) = UInt32
_unsigned(::Type{Float64}) = UInt64
function Base.issubnormal(x::Vec{N, T}) where {N, T<:FloatingTypes}
    y = reinterpret(Vec{N, _unsigned(T)}, x)
    (y & Base.exponent_mask(T) == 0) & (y & Base.significand_mask(T) != 0)
end

@inline Base.signbit(x::Vec{N, <:IntegerTypes}) where {N} = x < 0

@inline Base.leading_ones(x::Vec{<:Any, <:IntegerTypes})  = leading_zeros(~(x))
@inline Base.trailing_ones(x::Vec{<:Any, <:IntegerTypes}) = trailing_zeros(~(x))
@inline Base.count_zeros(x::Vec{<:Any, <:IntegerTypes}) = count_zeros(~(x))

@inline Base.isnan(v::Vec{<:Any, <:FloatingTypes}) = v != v
@inline Base.isfinite(v::Vec{<:Any, <:FloatingTypes}) = v - v == zero(v)
@inline Base.isinf(v::Vec{<:Any, <:FloatingTypes}) = !isnan(v) & !isfinite(v)
@inline Base.sign(v1::Vec{N,T}) where {N,T} =
    vifelse(v1 == zero(Vec{N,T}), zero(Vec{N,T}),
            vifelse(v1 < zero(Vec{N,T}), -one(Vec{N,T}), one(Vec{N,T})))

@inline Base.isnan(v::Vec{N, <:IntegerTypes}) where {N} = zero(Vec{N,Bool})
@inline Base.isfinite(v::Vec{N, <:IntegerTypes}) where {N} = one(Vec{N, Bool})
@inline Base.isinf(v::Vec{N, <:IntegerTypes}) where {N} = zero(Vec{N, Bool})


####################
# Binary operators #
####################

const BINARY_OPS = [
    (:+        , IntegerTypes  , LLVM.add)
    (:-        , IntegerTypes  , LLVM.sub)
    (:*        , IntegerTypes  , LLVM.mul)
    (:div      , UIntTypes     , LLVM.udiv)
    (:div      , IntTypes      , LLVM.sdiv)
    (:rem      , UIntTypes     , LLVM.urem)
    (:rem      , IntTypes      , LLVM.srem)

    (:+        , FloatingTypes , LLVM.fadd)
    (:-        , FloatingTypes , LLVM.fsub)
    (:*        , FloatingTypes , LLVM.fmul)
    (:^        , FloatingTypes , LLVM.pow)
    (:/        , FloatingTypes , LLVM.fdiv)
    (:rem      , FloatingTypes , LLVM.frem)
    (:min      , FloatingTypes , LLVM.minnum)
    (:max      , FloatingTypes , LLVM.maxnum)
    (:copysign , FloatingTypes , LLVM.copysign)

    (:~        , IntegerTypes  , LLVM.xor)
    (:&        , IntegerTypes  , LLVM.and)
    (:|        , IntegerTypes  , LLVM.or)
    (:⊻        , IntegerTypes  , LLVM.xor)

    (:(==)     , IntegerTypes  , LLVM.icmp_eq)
    (:(!=)     , IntegerTypes  , LLVM.icmp_ne)
    (:(>)      , IntTypes      , LLVM.icmp_sgt)
    (:(>=)     , IntTypes      , LLVM.icmp_sge)
    (:(<)      , IntTypes      , LLVM.icmp_slt)
    (:(<=)     , IntTypes      , LLVM.icmp_sle)
    (:(>)      , UIntTypes     , LLVM.icmp_ugt)
    (:(>=)     , UIntTypes     , LLVM.icmp_uge)
    (:(<)      , UIntTypes     , LLVM.icmp_ult)
    (:(<=)     , UIntTypes     , LLVM.icmp_ule)

    (:(==)     , FloatingTypes , LLVM.fcmp_oeq)
    (:(!=)     , FloatingTypes , LLVM.fcmp_une)
    (:(>)      , FloatingTypes , LLVM.fcmp_ogt)
    (:(>=)     , FloatingTypes , LLVM.fcmp_oge)
    (:(<)      , FloatingTypes , LLVM.fcmp_olt)
    (:(<=)     , FloatingTypes , LLVM.fcmp_ole)
]

for (op, constraint, llvmop) in BINARY_OPS
    @eval @inline function (Base.$op)(x::Vec{N, T}, y::Vec{N, T}) where {N, T <: $constraint}
        Vec($(llvmop)(x.data, y.data))
    end
end

# max min
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

# Sign
@inline Base.flipsign(v1::Vec{N,T}, v2::Vec{N,T}) where {N,T} =
    vifelse(signbit(v2), -v1, v1)
@inline Base.copysign(v1::Vec{N,T}, v2::Vec{N,T}) where {N,T<:IntTypes} =
    vifelse(signbit(v2), -abs(v1), abs(v1))
_signed(::Type{Float32}) = Int32
_signed(::Type{Float64}) = Int64
@inline Base.signbit(x::Vec{N, T}) where {N, T <:FloatingTypes} = 
    signbit(reinterpret(Vec{N, _signed(T)}, x))


# Bitshifts
# See https://github.com/JuliaLang/julia/blob/7426625b5c07b0d93110293246089a259a0a677d/src/intrinsics.cpp#L1179-L1196
@inline function shl_int(x::Vec{N, T1}, y::Vec{N, T2}) where {N, T1<:IntegerTypes, T2<:IntegerTypes}
    vifelse(y > sizeof(T1) * 8,
        zero(Vec{N, T1}),
        Vec(LLVM.shl(x.data, convert(Vec{N,T1}, y).data)))
end

@inline function lshr_int(x::Vec{N, T1}, y::Vec{N, T2}) where {N, T1<:IntegerTypes, T2<:IntegerTypes}
    vifelse(y > sizeof(T1) * 8,
        zero(Vec{N, T1}),
        Vec(LLVM.lshr(x.data, convert(Vec{N,T1}, y).data)))
end

@inline function ashr_int(x::Vec{N, T1}, y::Vec{N, T2}) where {N, T1<:IntegerTypes, T2<:IntegerTypes}
    vifelse(y > sizeof(T1) * 8,
            Vec(LLVM.ashr(x.data, Vec{N,T1}(sizeof(T1)*8-1).data)),
            Vec(LLVM.ashr(x.data, Vec{N,T1}(y).data)))
end

# See https://github.com/JuliaLang/julia/blob/a211abcdfacc05cb93c15774a59ce8961c16dac4/base/int.jl#L422-L435
@inline Base.:>>(x::Vec{N, <:IntTypes}, y::Vec{N, <:UIntTypes}) where {N} =
    ashr_int(x, y)
@inline Base.:>>(x::Vec{N, T1}, y::Vec{N, T2}) where {N, T1<:UIntTypes, T2<:UIntTypes} =
    lshr_int(x, y)
@inline Base.:<<(x::Vec{N, T1}, y::Vec{N, T2}) where {N, T1<:IntegerTypes, T2<:UIntTypes} =
    shl_int(x, y)
@inline Base.:>>>(x::Vec{N, T1}, y::Vec{N, T2}) where {N, T1<:IntegerTypes, T2<:UIntTypes} =
    lshr_int(x, y)

@inline unsigned(v::Vec{<:Any, <:UIntTypes}) = v
@inline unsigned(v::Vec{N, Int32}) where {N} = convert(Vec{N, UInt32}, v)
@inline unsigned(v::Vec{N, Int64}) where {N} = convert(Vec{N, UInt64}, v)

@inline Base.:>>(x::Vec{N, T1}, y::Vec{N, Int}) where {N, T1<:IntegerTypes} =
    vifelse(0 <= y, x >> unsigned(y), x << unsigned(-y))
@inline Base.:<<(x::Vec{N, T1}, y::Vec{N, Int}) where {N, T1<:IntegerTypes} =
    vifelse(0 <= y, x << unsigned(y), x >> unsigned(-y))
@inline Base.:>>>(x::Vec{N, T1}, y::Vec{N, Int}) where {N, T1<:IntegerTypes} =
    vifelse(0 <= y, x >>> unsigned(y), x << unsigned(-y))

# Check these!
#=
for v in (:<<, :>>, :>>>)
    @eval begin
        @inline Base.$v(x::Vec{N,T}, y::ScalarTypes) where {N, T} = $v(x, Vec{N,T}(y))
        @inline Base.$v(x::ScalarTypes, y::Vec{N,T}) where {N, T} = $v(Vec{N,T}(x), y)
        @inline Base.$v(x::Vec{N,T1}, y::Vec{N,T2}) where {N, T1<:IntegerTypes, T2<:IntegerTypes} =
            $v(x, convert(Vec{N, Int}, y))
    end
end
=#

# Vectorize binary functions
for (op, constraint) in [BINARY_OPS; 
        (:flipsign , ScalarTypes)
        (:copysign , ScalarTypes)
        (:signbit  , ScalarTypes)
        (:min      , IntegerTypes)
        (:max      , IntegerTypes)
        (:^        , IntegerTypes)
        (:<<       , IntegerTypes)
        (:>>       , IntegerTypes)
        (:>>>      , IntegerTypes)
    ]
    @eval @inline function (Base.$op)(x::T2, y::Vec{N, T}) where {N, T2<:ScalarTypes, T <: $constraint}
        Base.$op(Vec{N, T}(x), y)
    end
    @eval @inline function (Base.$op)(x::Vec{N, T}, y::T2) where {N, T2 <:ScalarTypes, T <: $constraint}
        Base.$op(x, Vec{N, T}(x))
    end
end

#####################
# Ternary operators #
#####################

@inline vifelse(v::Bool, v1::Vec{N, T}, v2::Vec{N, T}) where {N, T} = ifelse(v, v1, v2)
@inline vifelse(v::Bool, v1::Vec{N, T}, v2::ScalarTypes) where {N, T} = ifelse(v, v1, Vec{N,T}(v2))
@inline vifelse(v::Bool, v1::ScalarTypes, v2::Vec{N, T}) where {N, T} = ifelse(v, Vec{N,T}(v1), v2)

@inline vifelse(v::Bool, v1::T, v2::T) where {T} = ifelse(v, v1, v2)
@inline vifelse(v::Vec{N, Bool}, v1::Vec{N, T}, v2::Vec{N, T}) where {N, T} =
    Vec(LLVM.select(v.data, v1.data, v2.data))
@inline vifelse(v::Vec{N, Bool}, v1::T2, v2::Vec{N, T}) where {N, T, T2 <:ScalarTypes} = vifelse(v, Vec{N, T}(v1), v2)
@inline vifelse(v::Vec{N, Bool}, v1::Vec{N, T}, v2::T2) where {N, T, T2 <:ScalarTypes} = vifelse(v, v1, Vec{N, T}(v2))

# fma, muladd and vectorization of these
for (op, llvmop) in [(:fma, LLVM.fma), (:muladd, LLVM.fmuladd)]
    @eval begin
        @inline Base.$op(a::Vec{N, T}, b::Vec{N, T}, c::Vec{N, T}) where {N,T<:FloatingTypes} =
            Vec($llvmop(a.data, b.data, c.data))
        @inline Base.$op(s1::ScalarTypes, v2::Vec{N,T}, v3::Vec{N,T}) where {N,T<:FloatingTypes} =
            $op(Vec{N,T}(s1), v2, v3)
        @inline Base.$op(v1::Vec{N,T}, s2::ScalarTypes, v3::Vec{N,T}) where {N,T<:FloatingTypes} =
            $op(v1, Vec{N,T}(s2), v3)
        @inline Base.$op(s1::ScalarTypes, s2::ScalarTypes, v3::Vec{N,T}) where {N,T<:FloatingTypes} =
            $op(Vec{N,T}(s1), Vec{N,T}(s2), v3)
        @inline Base.$op(v1::Vec{N,T}, v2::Vec{N,T}, s3::ScalarTypes) where {N,T<:FloatingTypes} =
            $op(v1, v2, Vec{N,T}(s3))
        @inline Base.$op(s1::ScalarTypes, v2::Vec{N,T}, s3::ScalarTypes) where {N,T<:FloatingTypes} =
            $op(Vec{N,T}(s1), v2, Vec{N,T}(s3))
        @inline Base.$op(v1::Vec{N,T}, s2::ScalarTypes, s3::ScalarTypes) where {N,T<:FloatingTypes} =
            $op(v1, Vec{N,T}(s2), Vec{N,T}(s3))
    end
end


##############
# Reductions #
##############
const HORZ_REDUCTION_OPS = [
    (&   , IntegerTypes  , LLVM.reduce_and)
    (|   , IntegerTypes  , LLVM.reduce_and)
    (max , IntTypes      , LLVM.reduce_smax)
    (max , UIntTypes     , LLVM.reduce_umax)
    (max , FloatingTypes , LLVM.reduce_fmax)
    (min , IntTypes      , LLVM.reduce_smin)
    (min , UIntTypes     , LLVM.reduce_umin)
    (min , FloatingTypes , LLVM.reduce_fmin)
    (+   , IntegerTypes  , LLVM.reduce_add)
    (*   , IntegerTypes  , LLVM.reduce_mul)
    (+   , FloatingTypes , LLVM.reduce_fadd)
    (*   , FloatingTypes , LLVM.reduce_fmul)
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

############
# Shuffles #
############

@inline function shufflevector(x::Vec{N, T}, ::Val{I}) where {N, T, I}
    Vec(LLVM.shufflevector(x.data, Val(I)))
end
@inline function shufflevector(x::Vec{N, T}, y::Vec{N, T}, ::Val{I}) where {N, T, I}
    Vec(LLVM.shufflevector(x.data, y.data, Val(I)))
end

end
