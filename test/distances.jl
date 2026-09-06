@testset "Distances and two-sample metrics" begin
    @testset "one-dimensional Wasserstein distance" begin
        @test MCBench.wasserstein1d([0.0, 1.0], [1.0, 2.0]) ≈ 1.0
        @test MCBench.wasserstein1d([0.0, 1.0], [1.0, 2.0]; p=2) ≈ 1.0
        @test MCBench.wasserstein1d([0.0, 10.0], [1.0]; wa=[1.0, 0.0]) ≈ 1.0
        @test MCBench.wasserstein1d([1.0], [0.0, 2.0]; wb=[1.0, 1.0]) ≈ 1.0
        @test_throws ArgumentError MCBench.wasserstein1d(Float64[], [1.0])
        @test_throws DimensionMismatch MCBench.wasserstein1d(
            [1.0],
            [1.0];
            wa=[1.0, 2.0],
        )
    end

    @testset "projection helpers and sliced Wasserstein" begin
        Random.seed!(44)
        projection = MCBench.get_random_projection(4)
        @test length(projection) == 4
        @test norm(projection) ≈ 1.0
        @test MCBench.project_sample([1.0, 2.0], [0.0, 1.0]) == 2.0
        @test MCBench.project_samples([[1.0, 2.0], [3.0, 4.0]], [1.0, 0.0]) == [1.0, 3.0]

        first_sample = MCBench.make_dsv(collect(0.0:4.0))
        shifted = MCBench.make_dsv(collect(1.0:5.0))
        distance = MCBench.get_sliced_wasserstein_distance(first_sample, shifted; L=8, parallel=false)
        @test distance ≈ 1.0
    end

    @testset "MMD implementation" begin
        x = reshape([0.0, 1.0], 1, :)
        y = reshape([2.0, 3.0], 1, :)
        kernel = MCBench.GaussianKernel(0.5)

        @test kernel(0.0) == 1.0
        @test kernel([0.0, 2.0]) ≈ [1.0, exp(-1.0)]
        @test MCBench.compute_bandwidth(x, y) ≈ 2.5
        @test MCBench.get_mmd(x, y) ≈ MCBench.get_mmd(x, y; g=0.2)
        @test MCBench.get_mmd(10 .* x, 10 .* y) ≈ MCBench.get_mmd(x, y)
        @test MCBench.get_mmd(x, y; g=0.5) ≈ MCBench.get_mmd(y, x; g=0.5)
        @test isfinite(MCBench.get_mmd(x, y))
        @test_throws ArgumentError MCBench.get_mmd(zeros(1, 2), zeros(1, 2))
    end

    @testset "two-sample metric dispatch" begin
        testcase = standard_normal_testcase(1; info="Two-Sample-Test")
        first_sample = MCBench.make_dsv([-1.0, 0.0, 1.0])
        shifted = MCBench.make_dsv([1.0, 2.0, 3.0])

        wasserstein = MCBench.calc_metric(testcase, first_sample, shifted, MCBench.wasserstein_1d())
        mmd = MCBench.calc_metric(testcase, first_sample, shifted, MCBench.maximum_mean_discrepancy(0.0, 0))
        sliced = MCBench.calc_metric(testcase, first_sample, shifted, MCBench.sliced_wasserstein_distance(0.0, 0))
        chi_squared = MCBench.calc_metric(testcase, first_sample, shifted, MCBench.chi_squared())

        @test only(wasserstein).val ≈ 2.0
        @test isfinite(only(mmd).val)
        @test only(sliced).val ≈ 2.0
        @test only(chi_squared).val ≈ 6.0

        dispatched = MCBench.run_teststatistic(
            testcase,
            first_sample,
            shifted,
            MCBench.wasserstein_1d(),
            MCBench.IIDSampler(),
        )
        @test only(dispatched).val ≈ 2.0

        Random.seed!(45)
        sampled_dispatch = MCBench.run_teststatistic(
            testcase,
            MCBench.wasserstein_1d();
            n_steps=10,
        )
        @test length(sampled_dispatch) == 1
        @test isfinite(only(sampled_dispatch).val)
    end
end
