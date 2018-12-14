#
#module SIMDIntrinsics

const VE = Base.VecElement
const Vec{N, T} = NTuple{N, Base.VecElement{T}}

include("macro.jl")

const BoolTypes = Union{Bool}
const IntTypes = Union{Int8, Int16, Int32, Int64, Int128}
const UIntTypes = Union{UInt8, UInt16, UInt32, UInt64, UInt128}
const IntegerTypes = Union{BoolTypes, IntTypes, UIntTypes}
const FloatingTypes = Union{Float32, Float64} # Float16 support is non native in Julia
const ScalarTypes = Union{IntegerTypes, FloatingTypes}

struct LLVMBool end

const d = Dict{DataType, String}(
    LLVMBool=> "i1",
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

    Float16 => "half",
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
            s = """
            %3 = $($f) <$(N) x $(d[T])> %0, %1
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

@generated function extractelement(x::Vec{N, T}, i::IntTypes) where {N, T}
    s = """
    %3 = extractelement <$N x $(d[T])> %0, $(d[T]) %1
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
            s = """
            %2 = $($f) <$(N) x $(d[T1])> %0 to <$(N) x $(d[T2])>
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
        s = """
        %2 = $($f) <$(N) x $(d[T1])> %0 to <$(N) x $(d[T2])>
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
const FCMP_FLAGS = [false ,:oeq ,:ogt ,:oge ,:olt ,:ole ,:one ,:ord ,:ueq ,:ugt ,:uge ,:ult ,:ule ,:une ,:uno ,:true]

for (f, constraint, flags) in zip(("cmp", "fcmp"), (IntTypes, FloatingTypes), (CMP_FLAGS, FCMP_FLAGS))
    for flag in flags
        ftot = Symbol(string(f, "_", flag))
        @eval @generated function $ftot(x::Vec{N, T}, y::Vec{N, T}) where {N, T <: $constraint}
            fflag = $(QuoteNode(flag))
            s = """
            %3 = $($f) $(fflag) <$(N) x $(d[T])> %0, %1
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
        return :(
            $(Expr(:meta, :inline));
            ccall($(llvm_name($f, N, T)), llvmcall, Vec{N, T}, (Vec{N, T},), x)
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
        return :(
            $(Expr(:meta, :inline));
            ccall($(llvm_name($f, N, T)), llvmcall, Vec{N, T}, (Vec{N, T}, Vec{N, T}), x, y)
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
        return :(
            $(Expr(:meta, :inline));
            ccall($(llvm_name($f, N, T)), llvmcall, Vec{N, T}, (Vec{N, T},), x)
        )
    end
end





#end