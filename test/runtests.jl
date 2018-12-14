using SIMDIntrinsics
using Test

import SIMDIntrinsics: VE, LLVM

VE(v...) = Base.VecElement.(v)


@testset "binary operators" begin
    v = VE(1.0f0, 2.0f0, 3.0f0, 4.0f0)
    v2 = VE(5.0f0, 6.0f0, 3.0f0, 4.0f0)
    @test LLVM.fadd(v, v2) == VE(6.0f0, 8.0f0, 6.0f0, 8.0f0)
end
