# LLVM operations and intrinsics
module LLVM

# TODO masked loads and stores

import ..SIMDIntrinsics: VE, LVec, IntegerTypes, IntTypes, UIntTypes, FloatingTypes

const d = Dict{DataType, String}(
    Bool    => "i8",
    Int8    => "i8",
    Int16   => "i16",
    Int32   => "i32",
    Int64   => "i64",
    Int128  => "i128",

    UInt8   => "i8",
    UInt16  => "i16",
    UInt32  => "i32",
    UInt64  => "i64",
    UInt128 => "i128",

    #Float16 => "half",
    Float32 => "float",
    Float64 => "double",
    Ptr{Float64} => "i64"
)


# TODO: Clean up
suffix(N::Integer, ::Type{Ptr{T}}) where {T} = "v$(N)p0$(T<:IntegerTypes ? "i" : "f")$(8*sizeof(T))"
suffix(N::Integer, ::Type{T}) where {T} = "v$(N)$(T<:IntegerTypes ? "i" : "f")$(8*sizeof(T))"
llvm_name(llvmf, N, T) = string("llvm", ".", llvmf, ".", suffix(N, T))

#####################
# Binary operators  #
#####################

# Type preserving

# (signed, unsigned, float)
const BINARY_OPS = [
    (:add, :add, :fadd),
    (:sub, :sub, :fsub),
    (:mul, :mul, :fmul),
    (:sdiv, :udiv, :fdiv),
    (:srem, :urem, :frem),
    # Bitwise
    (:shl, :shl),
    (:ashr, :lshr),
    (:and, :and),
    (:or, :or,),
    (:xor, :xor),
]

for fs in BINARY_OPS
    for (f, constraint) in zip(fs, (IntTypes, UIntTypes, FloatingTypes))
        @eval @generated function $f(x::LVec{N, T}, y::LVec{N, T}) where {N, T <: $constraint}
            ff = $(QuoteNode(f))
            s = """
            %3 = $ff <$(N) x $(d[T])> %0, %1
            ret <$(N) x $(d[T])> %3
            """
            return :(
                $(Expr(:meta, :inline));
                Base.llvmcall($s, LVec{N, T}, Tuple{LVec{N, T}, LVec{N, T}}, x, y)
            )
        end
    end
end

################
# Load / store #
################

# TODO: Alignment
@generated function load(x::Type{LVec{N, T}}, ptr::Ptr{T}) where {N, T}
    s = """
    %ptr = inttoptr $(d[Int]) %0 to <$N x $(d[T])>*
    %res = load <$N x $(d[T])>, <$N x $(d[T])>* %ptr, align 8
    ret <$N x $(d[T])> %res
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, LVec{N, T}, Tuple{Ptr{T}}, ptr)
    )
end

@generated function store(x::LVec{N, T}, ptr::Ptr{T}) where {N, T}
    s = """
    %ptr = inttoptr $(d[Int]) %1 to <$N x $(d[T])>*
    store <$N x $(d[T])> %0, <$N x $(d[T])>* %ptr, align 8
    ret void
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, Cvoid, Tuple{LVec{N, T}, Ptr{T}}, x, ptr)
    )
end

####################
# Gather / Scatter #
####################

@generated function maskedgather(::Type{LVec{N, T}}, ptrs::Union{LVec{N, Ptr{T}}, LVec{N, Int}}) where {N, T}
    # TODO: Allow setting the mask
    # TODO: Allow setting the passthru
    decl = "declare <$N x $(d[T])> @llvm.masked.gather.$(suffix(N, T))(<$N x $(d[T])*>, i32, <$N x i1>, <$N x $(d[T])>)"
    mask = join(("i1 true" for i in 1:N), ", ")

    s = """
    %ptrs = inttoptr <$N x $(d[Int])> %0 to <$N x $(d[T])*>
    %res = call <$N x $(d[T])> @llvm.masked.gather.$(suffix(N, T))(<$N x $(d[T])*> %ptrs, i32 8, <$N x i1> <$mask>, <$N x $(d[T])> undef)
    ret <$N x $(d[T])> %res
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall(($decl, $s), LVec{N, T}, Tuple{typeof(ptrs)}, ptrs)
    )
end

@generated function maskedscatter(x::LVec{N, T}, ptrs::Union{LVec{N, Int}, LVec{N, Ptr{T}}}) where {N, T}
    # TODO: Allow setting the mask
    mask = join(("i1 true " for i in 1:N), ", ")
    decl = "declare <$N x $(d[T])> @llvm.masked.scatter.$(suffix(N, T))(<$N x $(d[T])>, <$N x $(d[T])*>, i32, <$N x i1>)"

    s = """
    %ptrs = inttoptr <$N x $(d[Int])> %1 to <$N x $(d[T])*>
    call <$N x $(d[T])> @llvm.masked.scatter.$(suffix(N, T))(<$N x $(d[T])> %0, <$N x $(d[T])*> %ptrs, i32 8, <$N x i1> <$mask>)
    ret void
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall(($decl, $s), Cvoid, Tuple{LVec{N, T}, typeof(ptrs)}, x, ptrs)
    )
end


######################
# LVector Operations #
######################

@generated function extractelement(x::LVec{N, T}, i::I) where {N, T, I <: IntTypes}
    s = """
    %3 = extractelement <$N x $(d[T])> %0, $(d[I]) %1
    ret $(d[T]) %3
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, T, Tuple{LVec{N, T}, $i}, x, i)
    )
end

@generated function insertelement(x::LVec{N, T}, v::T, i::IntTypes) where {N, T}
    s = """
    %4 = insertelement <$N x $(d[T])> %0, $(d[T]) %1, $(d[i]) %2
    ret <$N x $(d[T])> %4
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, LVec{N, T}, Tuple{LVec{N, T}, T, typeof(i)}, x, v, i)
    )
end

@generated function insertelement(x::LVec{N, T}, v::T, ::Val{i}) where {N, T, i}
    s = """
    %3 = insertelement <$N x $(d[T])> %0, $(d[T]) %1, $(d[Int]) $(i-1)
    ret <$N x $(d[T])> %3
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, LVec{N, T}, Tuple{LVec{N, T}, T}, x, v)
    )
end

@generated function shufflevector(x::LVec{N, T}, y::LVec{N, T}, ::Val{I}) where {N, T, I}
    # Assert I < 2N?
    shfl = join((string("i32 ", i) for i in I), ", ")
    s = """
    %res = shufflevector <$N x $(d[T])> %0, <$N x $(d[T])> %1, <$N x i32> <$shfl>
    ret <$N x $(d[T])> %res
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, LVec{N, T}, Tuple{LVec{N, T}, LVec{N, T}}, x, y)
    )
end

@generated function shufflevector(x::LVec{N, T}, ::Val{I}) where {N, T, I}
    # Assert I < N?
    shfl = join((string("i32 ", i) for i in I), ", ")
    s = """
    %res = shufflevector <$N x $(d[T])> %0, <$N x $(d[T])> undef, <$N x i32> <$shfl>
    ret <$N x $(d[T])> %res
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, LVec{N, T}, Tuple{LVec{N, T}}, x)
    )
end

@generated function constantvector(v::T, y::Type{LVec{N, T}}) where {N, T}
    s = """
    %2 = insertelement <$N x $(d[T])> undef, $(d[T]) %0, i32 0
    %res = shufflevector <$N x $(d[T])> %2, <$N x $(d[T])> undef, <$N x i32> zeroinitializer
    ret <$N x $(d[T])> %res
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, LVec{N, T}, Tuple{T}, v)
    )
end

#########################
# Conversion Operations #
#########################
# Conversions

const CONVERSION_OPS_SIZE_CHANGE_SAME_ELEMENTS = [
    ((:trunc, :trunc, :fptrunc), >),
    ((:zext, :zext,   :fpext),   <),
    ((:sext, :sext),             <),
]

for (fs, criteria) in CONVERSION_OPS_SIZE_CHANGE_SAME_ELEMENTS
    for (f, constraint) in zip(fs, (IntTypes, UIntTypes, FloatingTypes))
        @eval @generated function $f(::Type{LVec{N, T2}}, x::LVec{N, T1}) where {N, T1 <: $constraint, T2 <: $constraint}
            sT1, sT2 = sizeof(T1) * 8, sizeof(T2) * 8
            @assert $criteria(sT1, sT2) "size of conversion type ($T2: $sT2) must be $($criteria) than the element type ($T1: $sT1)"
            ff = $(QuoteNode(f))
            s = """
            %2 = $ff <$(N) x $(d[T1])> %0 to <$(N) x $(d[T2])>
            ret <$(N) x $(d[T2])> %2
            """
            return :(
                $(Expr(:meta, :inline));
                Base.llvmcall($s, LVec{N, T2}, Tuple{LVec{N, T1}}, x)
            )
        end
    end
end

const CONVERSION_TYPES = [
    (:fptoui, (FloatingTypes, UIntTypes)),
    (:fptosi, (FloatingTypes, IntTypes)),
    (:uitofp, (UIntTypes, FloatingTypes)),
    (:sitofp, (IntTypes, FloatingTypes)),
]

for (f, (from, to)) in CONVERSION_TYPES
    @eval @generated function $f(::Type{LVec{N, T2}}, x::LVec{N, T1}) where {N, T1 <: $from, T2 <: $to}
        ff = $(QuoteNode(f))
        s = """
        %2 = $ff <$(N) x $(d[T1])> %0 to <$(N) x $(d[T2])>
        ret <$(N) x $(d[T2])> %2
        """
        return :(
            $(Expr(:meta, :inline));
            Base.llvmcall($s, LVec{N, T2}, Tuple{LVec{N, T1}}, x)
        )
    end
end


###########
# Bitcast #
###########

@generated function bitcast(::Type{LVec{N1, T1}}, x::LVec{N2, T2}) where {N1, T1, N2, T2}
    sT1, sT2 = sizeof(T1) * 8 * N1, sizeof(T2) * 8 * N2
    @assert sT1 == sT2 "size of conversion type ($N1 x $T1: $sT1) must be equal to the vector type ($N2 x $T2: $sT2)"
    s = """
    %2 = bitcast <$(N2) x $(d[T2])> %0 to <$(N1) x $(d[T1])>
    ret <$(N1) x $(d[T1])> %2
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, LVec{N1, T1}, Tuple{LVec{N2, T2}}, x)
    )
end

@generated function bitcast(::Type{T1}, x::LVec{N2, T2}) where {T1, N2, T2}
    sT1, sT2 = sizeof(T1) * 8, sizeof(T2) * 8 * N2
    @assert sT1 == sT2 "size of conversion type ($T1: $sT1) must be equal to the vector type ($N2 x $T2: $sT2)"
    s = """
    %2 = bitcast <$(N2) x $(d[T2])> %0 to $(d[T1])
    ret $(d[T1]) %2
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, T1, Tuple{LVec{N2, T2}}, x)
    )
end


###############
# Comparisons #
###############

const S_CMP_FLAGS = [:eq ,:ne, :sgt ,:sge ,:slt ,:sle]
const U_CMP_FLAGS = [:eq ,:ne ,:ugt ,:uge ,:ult ,:ule]
const FCMP_FLAGS = [:false ,:oeq ,:ogt ,:oge ,:olt ,:ole ,:one ,:ord ,:ueq ,:ugt ,:uge ,:ult ,:ule ,:une ,:uno , :true]

for (f, constraint, flags) in zip(("icmp", "icmp", "fcmp"), (IntTypes, UIntTypes, FloatingTypes), (S_CMP_FLAGS, U_CMP_FLAGS, FCMP_FLAGS))
    for flag in flags
        ftot = Symbol(string(f, "_", flag))
        @eval @generated function $ftot(x::LVec{N, T}, y::LVec{N, T}) where {N, T <: $constraint}
            fflag = $(QuoteNode(flag))
            ff = $(QuoteNode(f))
            s = """
            %3 = $ff $(fflag) <$(N) x $(d[T])> %0, %1
            %4 = sext <$(N) x i1> %3 to <$(N) x i8>
            ret <$(N) x i8> %4
            """
            return :(
                $(Expr(:meta, :inline));
                Base.llvmcall($s, LVec{N, Bool}, Tuple{LVec{N, T}, LVec{N, T}}, x, y)
            )
        end
    end
end

####################
# Unary operators  #
####################

const UNARY_INTRINSICS = [
    :sqrt
    :sin
    :cos
    :exp
    :exp2
    :log
    :log10
    :log2
    :fabs
    :floor
    :ceil
    :rint
    :nearbyint
    :round
]

for f in UNARY_INTRINSICS
    @eval begin
    @generated function $(f)(x::LVec{N, T}) where {N, T <: FloatingTypes}
        ff = llvm_name($(QuoteNode(f)), N, T)
        return :(
            $(Expr(:meta, :inline));
            ccall($ff, llvmcall, LVec{N, T}, (LVec{N, T},), x)
        )
    end
    end
end

const BINARY_INTRINSICS = [
    :minnum,
    :maxnum,
    :minimum,
    :maximum,
    :copysign,
    :pow,
    :floor,
    :ceil,
    :trunc,
    :rint,
    :nearbyint,
    :round,
]

for f in BINARY_INTRINSICS
    @eval @generated function $(f)(x::LVec{N, T}, y::LVec{N, T}) where {N, T <: FloatingTypes}
        ff = llvm_name($(QuoteNode(f)), N, T)
        return :(
            $(Expr(:meta, :inline));
            ccall($ff, llvmcall, LVec{N, T}, (LVec{N, T}, LVec{N, T}), x, y)
        )
    end
end

# pow, powi
for (f, constraint) in [(:pow, FloatingTypes), (:powi, IntegerTypes)]
    @eval @generated function $(f)(x::LVec{N, T1}, y::T2) where {N, T1 <: FloatingTypes, T2 <: $constraint}
        ff = llvm_name($(QuoteNode(f)), N, T1)
        return :(
            $(Expr(:meta, :inline));
            ccall($ff, llvmcall, LVec{N, T1}, (LVec{N, T1}, T2), x, y)
        )
    end

end


####################
# Bit manipulation #
####################

const BITMANIPULATION_INTRINSICS = [
    :bitreverse,
    :bswap,
    :ctpop,
    :ctlz,
    :cttz,
    :fshl,
    :fshr,
]

for f in BITMANIPULATION_INTRINSICS
    @eval @generated function $(f)(x::LVec{N, T}) where {N, T <: IntegerTypes}
        ff = llvm_name($(QuoteNode(f)), N, T)
        return :(
            $(Expr(:meta, :inline));
            ccall($ff, llvmcall, LVec{N, T}, (LVec{N, T},), x)
        )
    end
end

@generated function or(x::LVec{N, T}) where {N, T <: IntegerTypes}
    ff = llvm_name(:xor, N, T)
    shfl = join((string(d[T], " ", -1) for i in 1:N), ", ")
    s = """
    %res = xor <$N x $(d[T])> %0, <$shfl>
    ret <$N x $(d[T])> %res
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, LVec{N, T}, Tuple{LVec{N, T}}, x)
    )
end


##########
# Select #
##########

@generated function select(cond::LVec{N, Bool}, x::LVec{N, T}, y::LVec{N, T}) where {N, T}
    s = """
    %cond = trunc <$(N) x i8> %0 to <$(N) x i1>
    %res = select <$N x i1> %cond, <$N x $(d[T])> %1, <$N x $(d[T])> %2
    ret <$N x $(d[T])> %res
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, LVec{N, T}, Tuple{LVec{N, Bool}, LVec{N, T}, LVec{N, T}}, cond, x, y)
    )
end

###########
# Fmuladd #
###########

const MULADD_INTRINSICS = [
    :fmuladd,
    :fma,
]

for f in MULADD_INTRINSICS
    @eval @generated function $(f)(a::LVec{N, T}, b::LVec{N, T}, c::LVec{N, T}) where {N, T}
        ff = llvm_name($(QuoteNode(f)), N, T)
        return :(
            $(Expr(:meta, :inline));
            ccall($ff, llvmcall, LVec{N, T}, (LVec{N, T}, LVec{N, T}, LVec{N, T}), a, b, c)
        )
    end
end

#########################
# Horizontal reductions #
#########################

const HORZ_REDUCTION_OPS = [
    (:and, :and)
    (:or, :or)
    (:mul, :mul)
    (:add, :add)
    (:smax, :umax, :fmax)
    (:smin, :umin, :fmin)
]

for fs in HORZ_REDUCTION_OPS
    for (f, constraint) in zip(fs, (IntTypes, UIntTypes, FloatingTypes))
        f_red = Symbol("reduce_", f)
        @eval @generated function $f_red(x::LVec{N, T}) where {N,T<:$constraint}
            ff = llvm_name(string("experimental.vector.reduce.", $(QuoteNode(f))), N, T)
            decl = "declare $(d[T]) @$ff(<$N x $(d[T])>)"
            s2 = """
            %res = call $(d[T]) @$ff(<4 x $(d[T])> %0)
            ret $(d[T]) %res
            """
            return quote
                Base.llvmcall($(decl, s2), T, Tuple{LVec{4, T},}, x)
            end
        end
    end
end

# The fadd and fmul reductions take an initializer
# LLVM docs say these are called ‘llvm.experimental.vector.reduce.v2.fmul.*``
# That seems to not be true (no v2 below)
for (f, neutral) in [(:fadd, "0.0"), (:fmul, "1.0")]
    f_red = Symbol("reduce_", f)
    @eval @generated function $f_red(x::LVec{N, T}) where {N,T<:FloatingTypes}
        ff = llvm_name(string("experimental.vector.reduce.", $(QuoteNode(f))), N, T)
        decl = "declare $(d[T]) @$ff($(d[T]), <$N x $(d[T])>)"
        s2 = """
        %res = call $(d[T]) @$ff($(d[T]) $($neutral), <4 x $(d[T])> %0)
        ret $(d[T]) %res
        """
        return quote
            Base.llvmcall($(decl, s2), T, Tuple{LVec{4, T},}, x)
        end
    end
end

end
