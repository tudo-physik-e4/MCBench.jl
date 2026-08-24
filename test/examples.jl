@testset "Runnable examples" begin
    runner_path = joinpath(@__DIR__, "..", "scripts", "run_examples.jl")
    @test Meta.parseall(read(runner_path, String)) isa Expr
    quickstart_path = joinpath(@__DIR__, "..", "examples", "quickstart.jl")
    @test Meta.parseall(read(quickstart_path, String)) isa Expr

    include(joinpath(@__DIR__, "..", "examples", "basic_metrics.jl"))
    basic_result = BasicMetricsExample.main(n_samples=200, seed=77, verbose=false)
    @test length(basic_result.mean) == 2
    @test length(basic_result.variance) == 2
    @test isfinite(basic_result.wasserstein)
    @test isfinite(basic_result.mmd)
    @test basic_result.reference_mean == [0.0, 0.0]

    include(joinpath(@__DIR__, "..", "examples", "end_to_end_benchmark.jl"))
    mktempdir() do dir
        result = EndToEndBenchmarkExample.main(
            output_dir=dir,
            n_repetitions=3,
            n_samples=30,
            seed=88,
            verbose=false,
        )

        @test size(result.iid_mean) == (2, 3)
        @test size(result.sampler_mean) == (2, 3)
        @test length(result.plot_files) == 6
        @test any(contains("Variance"), result.plot_files)
        @test all(isfile, result.plot_files)
        @test all(path -> filesize(path) > 0, result.plot_files)
    end
end
