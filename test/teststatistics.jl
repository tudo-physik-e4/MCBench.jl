@testset "Test-statistic persistence" begin
    @testset "line and file parsing" begin
        mktempdir() do dir
            path = write_lines(joinpath(dir, "statistics.txt"), ["[1.0, 2.0]", "[3.0, 4.0]"])
            @test MCBench.parse_teststatistic(path) == [1.0 3.0; 2.0 4.0]

            open(path, "r") do io
                @test MCBench.parseline(io) == [1.0, 2.0]
                @test MCBench.parseline(io) == [3.0, 4.0]
            end

            blank_path = write_lines(joinpath(dir, "blank.txt"), [""])
            open(blank_path, "r") do io
                @test isempty(MCBench.parseline(io))
            end
        end
    end

    @testset "IID and sampler statistic files" begin
        mktempdir() do dir
            cd(dir) do
                testcase = standard_normal_testcase(2; info="Persistence-Test")
                metric = MCBench.marginal_mean()

                Random.seed!(55)
                output_paths = MCBench.build_teststatistic(
                    testcase,
                    [metric];
                    n=4,
                    n_steps=24,
                    n_samples=12,
                    par=false,
                    clean=true,
                    use_sampler=false,
                )
                @test output_paths == [joinpath(
                    "teststatistics",
                    "Persistence-Test-Mean.txt",
                )]
                iid_values = MCBench.read_teststatistic(testcase, metric)
                @test size(iid_values) == (2, 4)
                @test all(isfinite, iid_values)

                MCBench.build_teststatistic(
                    testcase,
                    [metric];
                    n=2,
                    n_steps=24,
                    n_samples=12,
                    par=false,
                    clean=false,
                    use_sampler=false,
                )
                @test size(MCBench.read_teststatistic(testcase, metric)) == (2, 6)

                quantile_metric = MCBench.marginal_quantiles()
                quantile_paths = MCBench.build_teststatistic(
                    testcase,
                    [quantile_metric];
                    n=2,
                    n_steps=24,
                    n_samples=12,
                    par=false,
                    clean=true,
                    use_sampler=false,
                )
                @test quantile_paths == [joinpath(
                    "teststatistics",
                    "Persistence-Test-Quantiles-50-90-99.txt",
                )]
                @test size(MCBench.read_teststatistic(testcase, quantile_metric)) == (6, 2)

                source = MCBench.sample(testcase, 30)
                sampler = MCBench.DsvSampler([source]; info="Compared-Sampler")
                MCBench.build_teststatistic(
                    testcase,
                    [metric];
                    s=sampler,
                    n=3,
                    n_steps=30,
                    n_samples=15,
                    par=false,
                    clean=true,
                    use_sampler=false,
                )
                sampler_values = MCBench.read_teststatistic(testcase, metric, sampler)
                @test size(sampler_values) == (2, 3)
                @test all(isfinite, sampler_values)

                parallel_testcase = standard_normal_testcase(2; info="Parallel-Test")
                parallel_metrics = MCBench.TestMetric[
                    MCBench.marginal_mean(),
                    MCBench.marginal_variance(),
                ]
                MCBench.build_teststatistic(
                    parallel_testcase,
                    parallel_metrics;
                    n=2,
                    n_steps=20,
                    n_samples=0,
                    par=true,
                    clean=true,
                    use_sampler=false,
                )
                @test size(MCBench.read_teststatistic(parallel_testcase, parallel_metrics[1])) == (2, 2)
                @test size(MCBench.read_teststatistic(parallel_testcase, parallel_metrics[2])) == (2, 2)

                @test_throws ArgumentError MCBench.build_teststatistic(
                    testcase,
                    MCBench.TestMetric[];
                    n=1,
                )
            end
        end
    end

    @testset "sampler failures are not hidden" begin
        mktempdir() do dir
            cd(dir) do
                testcase = standard_normal_testcase(1; info="Failure-Test")
                sampler = FailingSampler("Failing")

                @test_throws ErrorException MCBench.build_teststatistic(
                    testcase,
                    [MCBench.marginal_mean()];
                    s=sampler,
                    n=1,
                    n_steps=10,
                    n_samples=0,
                    clean=true,
                )
            end
        end
    end
end
