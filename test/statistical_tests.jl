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

    @testset "one-sample reference-value test" begin
        centered_values = collect(1.0:5.0)
        centered = MCBench.reference_value_test(centered_values, 3.0)
        @test centered.statistic == 0.0
        @test centered.pvalue == 1.0
        @test centered.standard_error ≈ std(centered_values) / sqrt(5)
        @test centered.degrees_of_freedom == 4

        shifted = MCBench.reference_value_test(centered_values, 0.0)
        expected_statistic = mean(centered_values) / (std(centered_values) / sqrt(5))
        @test shifted.statistic ≈ expected_statistic
        @test shifted.pvalue ≈ 0.013235599563682695 rtol = 1e-12

        # A two-sided test is invariant to the sign of the difference and to
        # translating the data and reference by the same amount.
        opposite = MCBench.reference_value_test(-centered_values, 0.0)
        translated = MCBench.reference_value_test(centered_values .+ 10, 13.0)
        @test opposite.statistic ≈ -shifted.statistic
        @test opposite.pvalue ≈ shifted.pvalue
        @test translated.statistic ≈ centered.statistic atol = 1e-14
        @test translated.pvalue ≈ centered.pvalue

        # Deterministic repetitions use well-defined limiting results.
        exact_constant = MCBench.reference_value_test(fill(2.0, 4), 2.0)
        different_constant = MCBench.reference_value_test(fill(2.0, 4), 1.0)
        @test exact_constant.statistic == 0.0
        @test exact_constant.pvalue == 1.0
        @test different_constant.statistic == Inf
        @test different_constant.pvalue == 0.0
    end

    @testset "input validation" begin
        @test_throws ArgumentError MCBench.ks_test(Float64[], [1.0])
        @test_throws ArgumentError MCBench.ks_test([1.0], Float64[])
        @test_throws ArgumentError MCBench.ks_test([NaN], [1.0])
        @test_throws ArgumentError MCBench.ks_test([1.0], [Inf])
        @test_throws ArgumentError MCBench.reference_value_test([1.0], 1.0)
        @test_throws ArgumentError MCBench.reference_value_test([1.0, NaN], 1.0)
        @test_throws ArgumentError MCBench.reference_value_test([1.0, 2.0], Inf)
    end
end
