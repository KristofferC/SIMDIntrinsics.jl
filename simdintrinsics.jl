#
module SIMDIntrinsics

const VE{N, T} = NTuple{N, Base.VecElement{T}}

include("macro.jl")

const BoolTypes = Union{Bool}
const IntTypes = Union{Int8, Int16, Int32, Int64, Int128}
const UIntTypes = Union{UInt8, UInt16, UInt32, UInt64, UInt128}
const IntegerTypes = Union{BoolTypes, IntTypes, UIntTypes}
const FloatingTypes = Union{Float16, Float32, Float64}
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

# julia => ((signed, unsigned, float), types)
const BINARYOPS = Dict(
    :add => (("add", "add", "fadd"),   ScalarTypes),
    :sub => (("sub", "sub", "fsub"),   ScalarTypes),
    :mul => (("mul", "mul", "fmul"),   ScalarTypes),
    :div => (("sdiv", "udiv", "fdiv"), ScalarTypes),
    :rem => (("srem", "urem", "frem"), ScalarTypes),
    # Bitwise
    :shl => (("shl", "shl", ""),       IntegerTypes),
    :lshr => (("lshr", "lshr", ""),       IntegerTypes),
    :ashr => (("ashr", "ashr", ""),       IntegerTypes),
    :and => (("and", "and", ""),       IntegerTypes),
    :or  => (("or", "or", ""),         IntegerTypes),
    :xor => (("xor", "xor", ""),       IntegerTypes),
)
binop(f::Symbol, T) = BINARYOPS[f][1][T <: UIntTypes ? 1 : T <: IntegerTypes ? 2 : 3]

for (f, (_, types)) in BINARYOPS
    @eval begin
    @llvmcall function $f(x::VE{N, T}, y::VE{N, T})::VE{N, T} where {N, T <: $types}
        f = $(QuoteNode(f))
        """
        %3 = $(binop(f, T)) <$(N) x $(d[T])> %0, %1
        ret <$(N) x $(d[T])> %3
        """
    end
    end
end

#####################
# Vector Operations #
#####################

function extractelement(x::VE{N, T}, ::Val{I}) where {N, T, I}
    @assert 0 <= I < N
    s = "extractelement <$(N) x $(d[T])> %0, $(d[T]) %2"
    return :(
        Base.llvmcall($s, VE{N, T2}, Tuple{VE{N, T1}}, x, I)
    )
end



# Predicates
const BINARYOPS_PREDICATES = Dict(
    :cmp_eq => ("icmp eq", "", "fadd"),
    :sub => ("sub", "sub", "fsub"),
    :mul => ("mul", "mul", "fmul"),
    :mul => ("idiv", "udiv", "fdiv"),
)

#########################
# Conversion Operations #
#########################
# Conversions

@generated function zext(::Type{T2}, x::VE{N, T1}) where {N, T1 <: IntegerTypes, T2 <: IntegerTypes}
    s = """
    %2 = zext <$(N) x $(d[T1])> %0 to <$(N) x $(d[T2])>
    ret <$(N) x $(d[T2])> %2
    """
    return :(
        Base.llvmcall($s, VE{N, T2}, Tuple{VE{N, T1}}, x)
    )
end

####################
# Unary operators  #
####################

suffix(N::Integer, T::Type) = "v$(N)$(T<:IntegerTypes ? "i" : "f")$(8*sizeof(T))"
llvm_name(llvmf, N, T) = string("llvm", ".", llvmf, ".", suffix(N, T))

const UNARY_OPERATORS = [
    :sqrt => "sqrt",
]

for (f, llvmf) in UNARY_OPERATORS
    @eval begin
    @generated function $(f)(x::VE{N, T}) where {N, T <: ScalarTypes}
        :(ccall($(llvm_name($llvmf, N, T)), llvmcall, VE{N, T}, (VE{N, T},), x))
    end
    end
end

###########
# Special #
###########

# Promotion
############

end