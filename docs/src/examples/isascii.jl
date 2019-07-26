# #isascii

# Computes if a string only contains ASCII, 256 bits at a time
# TODO: Port to SIMDIntrinsics.SIMD

using SIMDIntrinsics.LLVM
function isascii_simd(s::String)
    len = sizeof(s)
    nwords = len >> 7
    _0x80 = LLVM.constantvector(0x80, LLVM.LVec{32, UInt8})
    p = pointer(s)
    i = 0
    GC.@preserve s for _ in 1:nwords
        comp = LLVM.constantvector(0x00, LLVM.LVec{32, UInt8})
        for _ in 1:4
            v = LLVM.load(LLVM.LVec{32, UInt8}, p + i)
            comp_i = LLVM.and(v, _0x80)
            comp = LLVM.add(comp, comp_i)
            i += 32
        end
        u = LLVM.bitcast(LLVM.LVec{4, UInt64}, comp)
        #' TODO: Is there a better way to check if any
        u1, u2, u3, u4 = LLVM.extractelement(u, 0), LLVM.extractelement(u, 1),
                         LLVM.extractelement(u, 2), LLVM.extractelement(u, 3)
        iszero(u1 | u2 | u3 | u4) || return false
    end
    #' Finish up the chunks
    for i = nwords*32*4+1:len
        @inbounds(codeunit(s, i)) >= 0x80 && return false
    end
    return true
end

# The inner loop:
# ```
# vpand   -96(%edx), %ymm0, %ymm1
# vpand   -64(%edx), %ymm0, %ymm2
# vpaddb  %ymm1, %ymm2, %ymm1
# vpand   -32(%edx), %ymm0, %ymm2
# vpand   (%edx), %ymm0, %ymm3
# vpaddb  %ymm3, %ymm2, %ymm2
# vpaddb  %ymm2, %ymm1, %ymm1
# vptest  %ymm1, %ymm1
# ```