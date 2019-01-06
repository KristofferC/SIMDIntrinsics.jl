
const __m256d = LVec{4, Float64}
@inline _mm256_add_ps(a::__m256d, b::__m256d) = LLVM.fadd(a, b)
@inline function _mm256_add_ss(a::__m256d, b::__m256d)
    v = LLVM.extractelement(a, 0) + LLVM.extractelement(b, 0)
    return LLVM.insertelement(a, v, 0)
end
