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
    end
end
