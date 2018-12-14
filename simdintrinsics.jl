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
const BINARYOPS = [
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

for fs in BINARYOPS
    for (f, constraint) in zip(fs, (IntTypes, UIntTypes, FloatingTypes))
        @eval @llvmcall function $f(x::Vec{N, T}, y::Vec{N, T})::Vec{N, T} where {N, T <: $constraint}
            f = $(QuoteNode(f))
            """
            %3 = $(f) <$(N) x $(d[T])> %0, %1
            ret <$(N) x $(d[T])> %3
            """
        end
    end
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

const CONVERSION_OPS_SIZE_CHANGE = [
    ((:trunc, :trunc, :fptrunc), <),
    ((:zext, :zext,   :fpext),   >),
    ((:sext, :sext),             >),
]

for (fs, criteria) in CONVERSION_OPS_SIZE_CHANGE
    for (f, constraint) in zip(fs, (IntTypes, UIntTypes, FloatingTypes))
        @eval @generated function $f(::Type{T2}, x::Vec{N, T1}) where {N, T1 <: $constraint, T2 <: $constraint}
            sT1, sT2 = sizeof(T2) * 8, sizeof(T1) * 8
            @assert $criteria(sT1, sT2) "size of conversion type ($T2: $sT2) must be $($criteria) than the element type ($T1: $sT2)"
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



# Predicates
const BINARYOPS_PREDICATES = Dict(
    :cmp_eq => ("icmp eq", "", "fadd"),
    :sub => ("sub", "sub", "fsub"),
    :mul => ("mul", "mul", "fmul"),
    :mul => ("idiv", "udiv", "fdiv"),
)


####################
# Unary operators  #
####################

suffix(N::Integer, T::Type) = "v$(N)$(T<:IntegerTypes ? "i" : "f")$(8*sizeof(T))"
llvm_name(llvmf, N, T) = string("llvm", ".", llvmf, ".", suffix(N, T))

const UNARY_OPERATORS = [
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

for f in UNARY_OPERATORS
    @eval begin
    @generated function $(f)(x::Vec{N, T}) where {N, T <: FloatingTypes}
        return :(
            $(Expr(:meta, :inline));
            ccall($(llvm_name($f, N, T)), llvmcall, Vec{N, T}, (Vec{N, T},), x)
        )
    end
    end
end

###########
# Special #
###########

# Promotion
############

#end