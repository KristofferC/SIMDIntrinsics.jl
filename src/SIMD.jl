# Imitate the API in SIMD.jl

module SIMD

import ..SIMDIntrinsics: LLVM, VE, LVec, ScalarTypes, IntegerTypes, IntTypes, UIntTypes, FloatingTypes, IndexTypes

struct Vec{N, T}
    data::LVec{N, T}
end
Vec(v::NTuple{N, T}) where {N, T <: ScalarTypes} = Vec(VE.(v))
Vec(v::Vararg{T, N}) where {N, T <: ScalarTypes} = Vec(v)

@inline function Base.promote_rule(::Type{Vec{N, T1}}, ::Type{Vec{N, T2}}) where {T1, T2, N}
    Vec{N, promote_type(T1, T2)}
end

@inline function Base.promote(v1::Vec{N, T1}, v2::Vec{N, T2}) where {T1, T2, N}
    return convert(promote_type(Vec{N, T1}, Vec{N, T2}), v1),
           convert(promote_type(Vec{N, T1}, Vec{N, T2}), v2)
end

# noop convert
@inline Base.convert(::Type{Vec{N,T}}, v::Vec{N,T}) where {N,T} = v

#  zext, sext, fpexpt
# trunc, trunc, fptrunc
# identity

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

Base.reinterpret(::Type{Vec{N1, T1}}, v::Vec) where {T1, N1} = Vec(LLVM.bitcast(LLVM.LVec{N1, T1}, v.data))
Base.reinterpret(::Type{T}, v::Vec) where {T} = Vec(LLVM.bitcast(T, v.data))


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

@inline Base.checkbounds(v::Vec, i::IntegerTypes) = i <= 0 || i > length(v.data) && Base.throw_boundserror(v, i)

function Base.getindex(v::Vec, i::IntegerTypes)
    @boundscheck checkbounds(v, i)
    return LLVM.extractelement(v.data, i-1)
end

function Base.setindex(x::Vec{N, T}, v, i::IntegerTypes) where {N, T}
    @boundscheck checkbounds(x, i)
    return LLVM.insertelement(x.data, T(v), i-1)
end


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
    (:/, IntegerTypes, LLVM.fdiv)
    (:rem, IntegerTypes, LLVM.frem)

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
    @eval function (Base.$op)(x::Vec{N, T}, y::Vec{N, T}) where {N, T <: $constraint}
        Vec($(llvmop)(x.data, y.data))
    end
end

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
    @eval function (Base.$op)(x::Vec{<:Any, <:$constraint})
        Vec($(llvmop)(x.data))
    end
end

Base.leading_ones(x::Vec{<:Any, <:IntegerTypes})  = leading_zeros(~(x))
Base.trailing_ones(x::Vec{<:Any, <:IntegerTypes}) = trailing_zeros(~(x))
Base.count_zeros(x::Vec{<:Any, <:IntegerTypes}) = count_zeros(~(x))

end