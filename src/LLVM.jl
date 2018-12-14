# LLVM operations and intrinsics
module LLVM

# TODO masked loads and stores

import ..SIMDIntrinsics: VE, Vec, IntegerTypes, IntTypes, UIntTypes, FloatingTypes

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
)

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
    (:lshr, :lshr),
    (:ashr, :ashr),
    (:and, :and),
    (:or, :or,),
    (:xor, :xor),
]

for fs in BINARY_OPS
    for (f, constraint) in zip(fs, (IntTypes, UIntTypes, FloatingTypes))
        @eval @generated function $f(x::Vec{N, T}, y::Vec{N, T}) where {N, T <: $constraint}
            ff = $(QuoteNode(f))
            s = """
            %3 = $ff <$(N) x $(d[T])> %0, %1
            ret <$(N) x $(d[T])> %3
            """
            return :(
                $(Expr(:meta, :inline));
                Base.llvmcall($s, Vec{N, T}, Tuple{Vec{N, T}, Vec{N, T}}, x, y)
            )
        end
    end
end

################
# Load / store #
################

# TODO: Alignment
@generated function load(x::Type{Vec{N, T}}, ptr::Ptr{T}) where {N, T}
    s = """
    %ptr = inttoptr $(d[Int]) %0 to <$N x $(d[T])>*
    %res = load <$N x $(d[T])>, <$N x $(d[T])>* %ptr, align 8
    ret <$N x $(d[T])> %res
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, Vec{N, T}, Tuple{Ptr{T}}, ptr)
    )
end

@generated function store(x::Vec{N, T}, ptr::Ptr{T}) where {N, T}
    s = """
    %ptr = inttoptr $(d[Int]) %1 to <$N x $(d[T])>*
    store <$N x $(d[T])> %0, <$N x $(d[T])>* %ptr, align 8
    ret void
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, Cvoid, Tuple{Vec{N, T}, Ptr{T}}, x, ptr)
    )
end


#####################
# Vector Operations #
#####################

@generated function extractelement(x::Vec{N, T}, i::I) where {N, T, I <: IntTypes}
    s = """
    %3 = extractelement <$N x $(d[T])> %0, $(d[I]) %1
    ret $(d[T]) %3
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, T, Tuple{Vec{N, T}, $i}, x, i)
    )
end

@generated function insertelement(x::Vec{N, T}, v::T, i::IntTypes) where {N, T}
    s = """
    %4 = insertelement <$N x $(d[T])> %0, $(d[T]) %1, $(d[i]) %2
    ret <$N x $(d[T])> %4
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, Vec{N, T}, Tuple{Vec{N, T}, T, $i}, x, v, i)
    )
end

@generated function shufflevector(x::Vec{N, T}, y::Vec{N, T}, ::Val{I}) where {N, T, I}
    shfl = join((string("i32 ", i) for i in I), ", ")
    s = """
    %res = shufflevector <$N x $(d[T])> %0, <$N x $(d[T])> %1, <$N x i32> <$shfl>
    ret <$N x $(d[T])> %res
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, Vec{N, T}, Tuple{Vec{N, T}, Vec{N, T}}, x, y)
    )
end

#########################
# Conversion Operations #
#########################
# Conversions

const CONVERSION_OPS_SIZE_CHANGE_SAME_ELEMENTS = [
    ((:trunc, :trunc, :fptrunc), <),
    ((:zext, :zext,   :fpext),   >),
    ((:sext, :sext),             >),
]

for (fs, criteria) in CONVERSION_OPS_SIZE_CHANGE_SAME_ELEMENTS
    for (f, constraint) in zip(fs, (IntTypes, UIntTypes, FloatingTypes))
        @eval @generated function $f(::Type{Vec{N, T2}}, x::Vec{N, T1}) where {N, T1 <: $constraint, T2 <: $constraint}
            sT1, sT2 = sizeof(T1) * 8, sizeof(T2) * 8
            @assert $criteria(sT1, sT2) "size of conversion type ($T2: $sT2) must be $($criteria) than the element type ($T1: $sT1)"
            ff = $(QuoteNode(f))
            s = """
            %2 = $ff <$(N) x $(d[T1])> %0 to <$(N) x $(d[T2])>
            ret <$(N) x $(d[T2])> %2
            """
            return :(
                $(Expr(:meta, :inline));
                Base.llvmcall($s, Vec{N, T2}, Tuple{Vec{N, T1}}, x)
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
    @eval @generated function $f(::Type{Vec{N, T2}}, x::Vec{N, T1}) where {N, T1 <: $from, T2 <: $to}
        ff = $(QuoteNode(f))
        s = """
        %2 = $ff <$(N) x $(d[T1])> %0 to <$(N) x $(d[T2])>
        ret <$(N) x $(d[T2])> %2
        """
        return :(
            $(Expr(:meta, :inline));
            Base.llvmcall($s, Vec{N, T2}, Tuple{Vec{N, T1}}, x)
        )
    end
end


###########
# Bitcast #
###########

@generated function bitcast(::Type{Vec{N, T2}}, x::Vec{N, T1}) where {N, T1, T2}
    sT2, sT1 = sizeof(T2) * 8, sizeof(T1) * 8
    @assert sT1 == sT2 "size of conversion type ($T2: $sT2) must be equal to the element type ($T1: $sT1)"
    s = """
    %2 = bitcast <$(N) x $(d[T1])> %0 to <$(N) x $(d[T2])>
    ret <$(N) x $(d[T2])> %2
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, Vec{N, T2}, Tuple{Vec{N, T1}}, x)
    )
end

@generated function bitcast(::Type{T2}, x::Vec{N, T1}) where {N, T1, T2}
    sT1, sT2 = sizeof(T2) * 8, sizeof(T1) * 8 * N
    @assert sT1 == sT2 "size of conversion type ($T2: $sT2) must be equal to the element type ($T1 x $N: $sT1)"
    s = """
    %2 = bitcast <$(N) x $(d[T1])> %0 to $(d[T2])
    ret $(d[T2]) %2
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, T2, Tuple{Vec{N, T1}}, x)
    )
end


###############
# Comparisons #
###############

const CMP_FLAGS = [:eq ,:ne ,:ugt ,:uge ,:ult ,:ule ,:sgt ,:sge ,:slt ,:sle]
const FCMP_FLAGS = [:false ,:oeq ,:ogt ,:oge ,:olt ,:ole ,:one ,:ord ,:ueq ,:ugt ,:uge ,:ult ,:ule ,:une ,:uno , :true]

for (f, constraint, flags) in zip(("cmp", "fcmp"), (IntTypes, FloatingTypes), (CMP_FLAGS, FCMP_FLAGS))
    for flag in flags
        ftot = Symbol(string(f, "_", flag))
        @eval @generated function $ftot(x::Vec{N, T}, y::Vec{N, T}) where {N, T <: $constraint}
            fflag = $(QuoteNode(flag))
            ff = $(QuoteNode(ftot))
            s = """
            %3 = $ff $(fflag) <$(N) x $(d[T])> %0, %1
            %4 = zext <$(N) x i1> %3 to <$(N) x i8>
            ret <$(N) x i8> %4
            """
            return :(
                $(Expr(:meta, :inline));
                Base.llvmcall($s, Vec{N, Bool}, Tuple{Vec{N, T}, Vec{N, T}}, x, y)
            )
        end
    end
end

####################
# Unary operators  #
####################

suffix(N::Integer, T::Type) = "v$(N)$(T<:IntegerTypes ? "i" : "f")$(8*sizeof(T))"
llvm_name(llvmf, N, T) = string("llvm", ".", llvmf, ".", suffix(N, T))

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
    @generated function $(f)(x::Vec{N, T}) where {N, T <: FloatingTypes}
        ff = llvm_name($(QuoteNode(f)), N, T)
        return :(
            $(Expr(:meta, :inline));
            ccall($ff, llvmcall, Vec{N, T}, (Vec{N, T},), x)
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
    :floor,
    :ceil,
    :trunc,
    :rint,
    :nearbyint,
    :round,
]

for f in BINARY_INTRINSICS
    @eval @generated function $(f)(x::Vec{N, T}, y::Vec{N, T}) where {N, T <: FloatingTypes}
        ff = llvm_name($(QuoteNode(f)), N, T)
        return :(
            $(Expr(:meta, :inline));
            ccall($ff, llvmcall, Vec{N, T}, (Vec{N, T}, Vec{N, T}), x, y)
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
    @eval @generated function $(f)(x::Vec{N, T}) where {N, T <: IntegerTypes}
        ff = llvm_name($(QuoteNode(f)), N, T)
        return :(
            $(Expr(:meta, :inline));
            ccall(ff, llvmcall, Vec{N, T}, (Vec{N, T},), x)
        )
    end
end

##########
# Select #
##########

@generated function select(cond::Vec{N, Bool}, x::Vec{N, T}, y::Vec{N, T}) where {N, T}
    s = """
    %cond = trunc <$(N) x i8> %0 to <$(N) x i1>
    %res = select <$N x i1> %cond, <$N x $(d[T])> %1, <$N x $(d[T])> %2
    ret <$N x $(d[T])> %res
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, Vec{N, T}, Tuple{Vec{N, Bool}, Vec{N, T}, Vec{N, T}}, cond, x, y)
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
    @eval @generated function $(f)(a::Vec{N, T}, b::Vec{N, T}, c::Vec{N, T}) where {N, T}
        ff = llvm_name($(QuoteNode(f)), N, T)
        return :(
            $(Expr(:meta, :inline));
            ccall($ff, llvmcall, Vec{N, T}, (Vec{N, T}, Vec{N, T}, Vec{N, T}), a, b, c)
        )
    end
end

end