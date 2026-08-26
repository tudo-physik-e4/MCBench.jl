@testset "Statistical comparison tests" begin
    @testset "known KS statistics and asymptotic p-values" begin
        x = collect(1.0:10.0)

        identical = MCBench.ks_test(x, copy(x))
        @test identical.statistic == 0.0
        @test identical.pvalue == 1.0

        half_overlap = MCBench.ks_test(x, collect(6.0:15.0))
        @test half_overlap.statistic ≈ 0.5
        @test half_overlap.pvalue ≈ 0.1640791977266521 rtol = 1e-12

        separated = MCBench.ks_test(x, collect(11.0:20.0))
        @test separated.statistic ≈ 1.0
        @test separated.pvalue ≈ 9.079985952496152e-5 rtol = 1e-12

        # This case checks the effective sample size nₓnᵧ/(nₓ+nᵧ) for unequal inputs.
        unequal = MCBench.ks_test(
            [1.0, 2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        @test unequal.statistic ≈ 2 / 3
        @test unequal.pvalue ≈ 0.23649007143707618 rtol = 1e-12

        # A strictly increasing transformation must not change either result.
        transformed = MCBench.ks_test(exp.(x), exp.(collect(6.0:15.0)))
        @test transformed.statistic ≈ half_overlap.statistic
        @test transformed.pvalue ≈ half_overlap.pvalue
    end

    @testset "continuous distribution calibration" begin
        rng = MersenneTwister(20260825)

        # Under the null, p-values should be approximately uniform for any
        # continuous distribution. Broad bounds keep this statistical check
        # robust while still detecting badly scaled or one-sided p-values.
        for distribution in (Normal(), Uniform(-2, 2), Exponential())
            null_pvalues = [
                MCBench.ks_test(
                    rand(rng, distribution, 100),
                    rand(rng, distribution, 100),
                ).pvalue
                for _ in 1:100
            ]
            @test 0.35 < mean(null_pvalues) < 0.65
            @test mean(null_pvalues .< 0.05) < 0.15
        end

        shifted_pvalues = [
            MCBench.ks_test(
                rand(rng, Normal(), 100),
                rand(rng, Normal(0.75, 1), 100),
            ).pvalue
            for _ in 1:100
        ]
        @test mean(shifted_pvalues .< 0.05) > 0.9
        @test median(shifted_pvalues) < 1e-3
    end

    @testset "input validation" begin
        @test_throws ArgumentError MCBench.ks_test(Float64[], [1.0])
        @test_throws ArgumentError MCBench.ks_test([1.0], Float64[])
        @test_throws ArgumentError MCBench.ks_test([NaN], [1.0])
        @test_throws ArgumentError MCBench.ks_test([1.0], [Inf])
    end
end
