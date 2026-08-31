@testset "Test cases and sampling" begin
    @testset "constructors" begin
        testcase = standard_normal_testcase(2; info="Constructor-Test")
        @test testcase.dim == 2
        @test testcase.info == "Constructor-Test"
        @test testcase.f isa MvNormal
        @test DensityInterface.DensityKind(testcase) isa DensityInterface.IsDensity

        default_bounds = MCBench.Testcases(Normal(2, 3), 1, "Default-Bounds")
        @test default_bounds.dim == 1
        @test default_bounds.info == "Default-Bounds"
        @test default_bounds.bounds isa NamedTupleDist
        @test isempty(default_bounds.reference_values)
        @test_throws ArgumentError MCBench.Testcases(Normal(), 0, "Invalid")
    end


    @testset "reference values" begin
        distribution = MvNormal(zeros(2), Matrix{Float64}(I, 2, 2))
        bounds = NamedTupleDist(x=fill(-5..5, 2))
        testcase = MCBench.Testcases(
            distribution,
            bounds,
            2,
            "References";
            reference_values=(
                marginal_mean=0.0,
                marginal_variance=[1.0, 2.0],
                marginal_skewness=[0.0, missing],
                marginal_mode=missing,
                marginal_quantiles=metric -> fill(4.2, 2 * length(metric.probabilities)),
                wasserstein_1d=0.0,
            ),
        )

        @test testcase.reference_values.marginal_mean == 0.0
        @test MCBench.reference_values(testcase, MCBench.marginal_mean()) == [0.0, 0.0]
        @test MCBench.reference_values(testcase, MCBench.marginal_variance()) == [1.0, 2.0]
        @test isequal(
            MCBench.reference_values(testcase, MCBench.marginal_skewness()),
            [0.0, missing],
        )
        @test isequal(
            MCBench.reference_values(testcase, MCBench.marginal_mode()),
            [missing, missing],
        )
        @test MCBench.reference_values(
            testcase,
            MCBench.marginal_quantiles([0.25, 0.75]),
        ) == fill(4.2, 4)
        @test MCBench.reference_values(testcase, MCBench.wasserstein_1d()) == [0.0, 0.0]
        @test isnothing(MCBench.reference_values(testcase, MCBench.global_mode()))

        wrong_dimension = MCBench.Testcases(
            distribution,
            bounds,
            2,
            "Wrong-References";
            reference_values=(marginal_mean=[0.0],),
        )
        @test_throws DimensionMismatch MCBench.reference_values(
            wrong_dimension,
            MCBench.marginal_mean(),
        )
        @test_throws ArgumentError MCBench.Testcases(
            distribution,
            bounds,
            2,
            "Invalid-References";
            reference_values=(marginal_mean="zero",),
        )
        @test_throws ArgumentError MCBench.Testcases(
            distribution,
            bounds,
            2,
            "Nonfinite-References";
            reference_values=(marginal_mean=Inf,),
        )
        @test_throws ArgumentError MCBench.Testcases(
            distribution,
            bounds,
            2,
            "Invalid-Partial-References";
            reference_values=(marginal_mean=[0.0, "unknown"],),
        )
        @test_throws ArgumentError MCBench.Testcases(
            distribution,
            bounds,
            2,
            "Nonfinite-Partial-References";
            reference_values=(marginal_mean=[missing, Inf],),
        )

        @test MCBench.reference_values(
            MCBench.normal_3d_uncorrelated,
            MCBench.marginal_mean(),
        ) == zeros(3)
        normal_quantiles = MCBench.reference_values(
            MCBench.normal_3d_uncorrelated,
            MCBench.marginal_quantiles([0.5, 0.9]),
        )
        @test normal_quantiles[1:3] ≈ zeros(3) atol=1e-14
        @test normal_quantiles[4:6] ≈ fill(quantile(Normal(), 0.9), 3)
        @test isnothing(MCBench.reference_values(
            MCBench.normal_3d_uncorrelated,
            MCBench.sliced_wasserstein_distance(),
        ))
        @test isnothing(MCBench.reference_values(
            MCBench.normal_3d_uncorrelated,
            MCBench.maximum_mean_discrepancy(),
        ))
        @test isnothing(MCBench.reference_values(
            MCBench.cauchy_1d,
            MCBench.marginal_mean(),
        ))
    end

    @testset "univariate distribution" begin
        Random.seed!(11)
        testcase = standard_normal_testcase(1)
        samples = MCBench.sample(testcase, 32)
        values = vec(sample_matrix(samples))

        @test length(samples) == 32
        @test size(sample_matrix(samples)) == (1, 32)
        @test samples.weight == ones(32)
        @test samples.logd ≈ logpdf.(Ref(testcase.f), values)
    end

    @testset "multivariate distribution and IID sampler" begin
        Random.seed!(12)
        testcase = standard_normal_testcase(3)
        direct_samples = MCBench.sample(testcase; n_steps=24)
        sampler_samples = MCBench.sample(testcase, MCBench.IIDSampler(24, "Small-IID"); n_steps=24)

        @test size(sample_matrix(direct_samples)) == (3, 24)
        @test size(sample_matrix(sampler_samples)) == (3, 24)
        @test direct_samples.logd ≈ logpdf(testcase.f, sample_matrix(direct_samples))
        @test sampler_samples.logd ≈ logpdf(testcase.f, sample_matrix(sampler_samples))
    end

    @testset "custom Target interface" begin
        target = DeterministicTarget(2)
        bounds = NamedTupleDist(x=fill(-5..5, 2))
        testcase = MCBench.Testcases(target, bounds, 2, "Deterministic")
        samples = MCBench.sample(testcase, 5)

        @test sample_matrix(samples) == repeat([1.0, 2.0], 1, 5)
        @test samples.logd == zeros(5)
    end

    @testset "sample-backed testcase" begin
        source = MCBench.make_dsv([0.0 1.0 2.0; 2.0 1.0 0.0], zeros(3))
        sampler = MCBench.DsvSampler([source]; info="Fixture-DSV")
        testcase = MCBench.DsvTestcase(sampler; info="DSV-Testcase")

        @test testcase.dim == 2
        @test testcase.info == "DSV-Testcase"
        @test length(MCBench.sample(testcase, 2)) == 2
        @test length(MCBench.sample(testcase; n_steps=2)) == 2
        @test length(MCBench.sample(testcase, MCBench.IIDSampler(); n_steps=2)) == 2

        referenced = MCBench.DsvTestcase(
            sampler;
            info="Referenced-DSV",
            reference_values=(marginal_mean=[1.0, 1.0],),
        )
        @test MCBench.reference_values(referenced, MCBench.marginal_mean()) == [1.0, 1.0]
    end
end
