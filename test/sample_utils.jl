@testset "Density sample utilities" begin
    @testset "BAT-shaped sample coordinates" begin
        shaped_samples = DensitySampleVector((
            [(x=[1.0, 2.0],), (x=[3.0, 4.0],)],
            zeros(2),
            ones(Int, 2),
            fill(nothing, 2),
            fill(nothing, 2),
        ))
        testcase = standard_normal_testcase(2; info="BAT-Shaped-Samples")
        means = MCBench.calc_metric(
            testcase,
            shaped_samples,
            MCBench.marginal_mean(),
        )
        @test getproperty.(means, :val) == [2.0, 3.0]
    end

    @testset "make_dsv overloads" begin
        matrix_values = [1.0 2.0 3.0; 4.0 5.0 6.0]
        matrix_dsv = MCBench.make_dsv(matrix_values, [-1.0, -2.0, -3.0]; weights=[1.0, 2.0, 1.0])
        default_matrix_dsv = MCBench.make_dsv(matrix_values)
        vector_dsv = MCBench.make_dsv([1.0, 2.0, 3.0])
        vectors_dsv = MCBench.make_dsv([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

        @test sample_matrix(matrix_dsv) == matrix_values
        @test matrix_dsv.logd == [-1.0, -2.0, -3.0]
        @test matrix_dsv.weight == [1.0, 2.0, 1.0]
        @test default_matrix_dsv.logd == ones(3)
        @test sample_matrix(vector_dsv) == reshape([1.0, 2.0, 3.0], 1, :)
        @test sample_matrix(vectors_dsv) == matrix_values
        @test vector_dsv.logd == ones(3)
        @test_throws DimensionMismatch MCBench.make_dsv(
            matrix_values,
            [-1.0, -2.0],
        )
        @test_throws DimensionMismatch MCBench.make_dsv(
            [[1.0], [2.0, 3.0]],
        )
    end

    @testset "weights and effective sample size" begin
        unweighted = MCBench.make_dsv([1.0, 2.0, 3.0, 4.0])
        weighted = MCBench.make_dsv([1.0, 2.0, 3.0], weights=[1.0, 2.0, 1.0])

        @test !MCBench.is_weighted(unweighted)
        @test MCBench.is_weighted(weighted)
        @test MCBench.get_effective_sample_size(unweighted) ≈ 4.0
        @test MCBench.get_effective_sample_size(unweighted, 1) ≈ 4.0
        @test MCBench.get_effective_sample_size(unweighted, MCBench.IIDSampler()) ≈ 4.0
    end

    @testset "condensing and resampling" begin
        repeated = MCBench.make_dsv([1.0, 1.0, 2.0, 2.0, 2.0])
        condensed = MCBench.condense_dsv(repeated)

        @test vec(sample_matrix(condensed)) == [1.0, 2.0]
        @test condensed.weight == [2, 3]
        @test_throws ArgumentError MCBench.condense_dsv(condensed)

        Random.seed!(22)
        @test length(MCBench.resample_dsv(repeated, 7)) == 7
        @test length(MCBench.resample_dsv_to_ess(repeated)) == 5
        @test length(MCBench.resample_dsv_to_ess(repeated, MCBench.IIDSampler())) == 5
    end

    @testset "two-sample preparation" begin
        first_sample = MCBench.make_dsv(collect(1.0:5.0))
        second_sample = MCBench.make_dsv(collect(1.0:3.0))
        same_first, same_second = MCBench.prepare_twosample_dsv(first_sample, first_sample)

        @test same_first === first_sample
        @test same_second === first_sample

        redirected = @test_logs (:warn,) MCBench.prepare_twosample_dsv(
            first_sample,
            second_sample,
        )
        @test length(Base.first(redirected)) == 3
        @test length(last(redirected)) == 3

        requested = @test_logs (:warn,) MCBench.prepare_twosample_dsv(
            first_sample,
            second_sample;
            N=2,
        )
        @test length(Base.first(requested)) == 2
        @test length(last(requested)) == 2

        capped = @test_logs (:warn,) MCBench.prepare_twosample_dsv(
            first_sample,
            first_sample;
            N=20,
        )
        @test length(Base.first(capped)) == 5
        @test length(last(capped)) == 5
        @test_throws ArgumentError MCBench.prepare_twosample_dsv(
            first_sample,
            second_sample;
            N=-1,
        )
    end
end
