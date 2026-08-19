@testset "Runnable examples" begin
    include(joinpath(@__DIR__, "..", "examples", "basic_metrics.jl"))
    basic_result = BasicMetricsExample.main(n_samples=200, seed=77, verbose=false)
    @test length(basic_result.mean) == 2
    @test length(basic_result.variance) == 2
    @test isfinite(basic_result.wasserstein)
    @test isfinite(basic_result.mmd)

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
        @test all(isfile, result.plot_files)
        @test all(path -> filesize(path) > 0, result.plot_files)
    end
end
