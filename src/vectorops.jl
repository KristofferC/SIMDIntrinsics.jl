using Base: Slice, ScalarIndex

"""
    ContiguousSubArray{T,N,P,I,L}
Like `Base.FastContiguousSubArray` but without requirement for linear
indexing (i.e., type parameter `L` can be `false`).

# Examples

```
julia> A = view(ones(5, 5), :, [1,3]);

julia> A isa Base.FastContiguousSubArray
false

julia> A isa SIMD.ContiguousSubArray
true
```
"""
ContiguousSubArray{T,N,P,
                   I<:Union{Tuple{Union{Slice, AbstractUnitRange}, Vararg{Any}},
                            Tuple{Vararg{ScalarIndex}}},
                   L} = SubArray{T,N,P,I,L}

"""
    ContiguousArray{T,N}

Array types with contiguous first dimension.
"""
ContiguousArray{T,N} = Union{DenseArray{T,N}, ContiguousSubArray{T,N}}

"""
    FastContiguousArray{T,N}

This is the type of arrays that `pointer(A, i)` works.
"""
FastContiguousArray{T,N} = Union{DenseArray{T,N}, Base.FastContiguousSubArray{T,N}}
# https://github.com/eschnett/SIMD.jl/pull/40#discussion_r254131184
# https://github.com/JuliaArrays/MappedArrays.jl/pull/24#issuecomment-460568978

export VecRange

"""
    VecRange{N}(i::Int)
    
Analogous to `UnitRange` but for loading SIMD vector of width `N` at
index `i`.

# Examples

```jldoctest
julia> xs = ones(4);
julia> xs[VecRange{4}(1)]  # calls `vload(Vec{4,Float64}, xs, 1)`
<4 x Float64>[1.0, 1.0, 1.0, 1.0]
```
"""
struct VecRange{N}
    i::Int
end

@inline Base.length(idx::VecRange{N}) where {N} = N
@inline Base.first(idx::VecRange) = idx.i
@inline Base.last(idx::VecRange) = idx.i + length(idx) - 1

@inline Base.:+(idx::VecRange{N}, j::Integer) where N = VecRange{N}(idx.i + j)
@inline Base.:+(j::Integer, idx::VecRange{N}) where N = VecRange{N}(idx.i + j)
@inline Base.:-(idx::VecRange{N}, j::Integer) where N = VecRange{N}(idx.i - j)


@inline vload(::Type{Vec{N, T}}, ptr::Ptr{T}) where {N, T} = Vec(LLVM.load(LLVM.LVec{N, T}, ptr))
@inline function vload(::Type{Vec{N, T}}, a::FastContiguousArray{T,1}, i::Integer) where {N, T}
    @boundscheck checkbounds(a, i + N - 1)
    GC.@preserve a begin
        return vload(Vec{N, T}, pointer(a, i))
    end
end
@propagate_inbounds Base.getindex(a::FastContiguousArray{T,1}, idx::VecRange{N}) where {N,T} =
    vload(Vec{N,T}, a, idx.i)

@inline vstore(x::Vec{N, T}, ptr::Ptr{T}) where {N, T} = LLVM.store(x.data, ptr)
@inline function vstore(x::Vec{N, T}, a::FastContiguousArray{T,1}, i::Integer) where {N, T}
    @boundscheck checkbounds(a, i + N - 1)
    GC.@preserve a begin
        vstore(x, pointer(a, i))
    end
    return a
end
@propagate_inbounds Base.setindex!(a::FastContiguousArray{T,1}, v::Vec{N, T}, idx::VecRange{N}) where {N,T} =
    vstore(v, a, idx.i)

function valloc(::Type{T}, N::Int, sz::Int) where T
    @assert N > 0
    @assert sz >= 0
    # We use padding to align the address of the first element, and
    # also to ensure that we can access past the last element up to
    # the next full vector width
    padding = N-1 + mod(-sz, N)
    mem = Vector{T}(undef, sz + padding)
    addr = Int(pointer(mem))
    off = mod(-addr, N * sizeof(T))
    @assert mod(off, sizeof(T)) == 0
    off = fld(off, sizeof(T))
    @assert 0 <= off <= padding
    res = view(mem, off+1 : off+sz)
    addr2 = Int(pointer(res))
    @assert mod(addr2, N * sizeof(T)) == 0
    res
end

function valloc(f, ::Type{T}, N::Int, sz::Int) where T
    mem = valloc(T, N, sz)
    @inbounds for i in 1:sz
        mem[i] = f(i)
    end
    mem
end

@inline function _get_vec_pointers(a, idx::Vec{N, Int}) where {N}
    p = Vec{N, Int}(Int(pointer(a)))
    ptrs = p + (idx - 1) * sizeof(eltype(a))
end

# Have to be careful with optional arguments and @boundscheck,
# see https://github.com/JuliaLang/julia/issues/30411,
# therefore use @propagate_inbounds
@propagate_inbounds function vgather(a::FastContiguousArray{T,1}, idx::Vec{N, Int}, mask::Vec{N,Bool}=one(Vec{N,Bool})) where {N, T<:ScalarTypes}
    @boundscheck for i in 1:N
        checkbounds(a, @inbounds idx[i])
    end
    GC.@preserve a begin
        ptrs = _get_vec_pointers(a, idx)
        return Vec(LLVM.maskedgather(LLVM.LVec{N, T}, ptrs.data, mask.data))
    end
end
@propagate_inbounds Base.getindex(a::FastContiguousArray{T,1}, idx::Vec{N,Int}) where {N,T} =
    vgather(a, idx)

@propagate_inbounds function vscatter(x::Vec{N,T}, a::FastContiguousArray{T,1},
        idx::Vec{N, Int}, mask::Vec{N,Bool}=one(Vec{N, Bool})) where {N, T<:ScalarTypes}
    @boundscheck for i in 1:N
        checkbounds(a, @inbounds idx[i])
    end
    GC.@preserve a begin
        ptrs = _get_vec_pointers(a, idx)
        LLVM.maskedscatter(x.data, ptrs.data, mask.data)
    end
    return
end
@propagate_inbounds Base.setindex!(a::FastContiguousArray{T,1}, v::Vec{N,T}, idx::Vec{N,Int}) where {N, T} =
    vscatter(v, a, idx)
