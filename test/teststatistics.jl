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

    @testset "ESS-aware sampler batching" begin
        mktempdir() do dir
            cd(dir) do
                testcase = standard_normal_testcase(1; info="ESS-Batching-Test")
                count_metric = SampleCountMetric()
                two_sample_metric = TwoSampleCountMetric()

                # Each complete sampler draw is preserved. Its conservative
                # ESS of two sets the size of that repetition's IID samples.
                sampler = FixedEffectiveSampleSizeSampler(7, 2.0, 0, "ESS-Sampler")
                MCBench.build_teststatistic(
                    testcase,
                    MCBench.TestMetric[count_metric, two_sample_metric];
                    s=sampler,
                    n=2,
                    n_steps=7,
                    n_samples=5,
                    unweight=true,
                    use_sampler=false,
                    ess_pilot_runs=1,
                    par=false,
                    clean=true,
                )

                sampler_counts = MCBench.read_teststatistic(
                    testcase,
                    count_metric,
                    sampler,
                )
                iid_counts = MCBench.read_teststatistic(testcase, count_metric)
                sampler_two_sample_counts = MCBench.read_teststatistic(
                    testcase,
                    two_sample_metric,
                    sampler,
                )
                iid_two_sample_counts = MCBench.read_teststatistic(
                    testcase,
                    two_sample_metric,
                )

                @test sampler_counts == fill(7.0, 1, 2)
                @test iid_counts == fill(2.0, 1, 2)
                @test sampler_two_sample_counts == fill(2_007.0, 1, 2)
                @test iid_two_sample_counts == fill(2_002.0, 1, 2)
                @test MCBench.read_effective_sample_sizes(testcase, sampler) == [2, 2]
                @test MCBench.read_matched_iid_sample_size(testcase, sampler) == 2
                @test sampler.draw_count == 3

                # Appending extends sampler, IID, and ESS outputs together.
                MCBench.build_teststatistic(
                    testcase,
                    [count_metric];
                    s=sampler,
                    n=1,
                    n_steps=7,
                    n_samples=5,
                    unweight=true,
                    use_sampler=false,
                    ess_pilot_runs=1,
                    par=false,
                    clean=false,
                )
                @test size(MCBench.read_teststatistic(
                    testcase,
                    count_metric,
                    sampler,
                )) == (1, 3)
                @test size(MCBench.read_teststatistic(testcase, count_metric)) == (1, 3)
                @test MCBench.read_effective_sample_sizes(testcase, sampler) == [2, 2, 2]
                @test sampler.draw_count == 4

                # Disabling unweighting restores raw-sample batching.
                raw_sampler = FixedEffectiveSampleSizeSampler(7, 2.0, 0, "Raw-Sampler")
                MCBench.build_teststatistic(
                    testcase,
                    [count_metric];
                    s=raw_sampler,
                    n=2,
                    n_steps=7,
                    n_samples=5,
                    unweight=false,
                    use_sampler=false,
                    par=false,
                    clean=true,
                )

                raw_values = MCBench.read_teststatistic(
                    testcase,
                    count_metric,
                    raw_sampler,
                )
                @test raw_values == fill(5.0, 1, 2)
                @test raw_sampler.draw_count == 2

                # Do not silently append an ESS-matched IID distribution to a
                # legacy baseline that used a different sample size.
                legacy_testcase = standard_normal_testcase(
                    1;
                    info="Legacy-IID-Baseline-Test",
                )
                MCBench.build_teststatistic(
                    legacy_testcase,
                    [count_metric];
                    n=1,
                    n_steps=5,
                    n_samples=5,
                    use_sampler=false,
                    par=false,
                    clean=true,
                )
                legacy_sampler = FixedEffectiveSampleSizeSampler(
                    7,
                    2.0,
                    0,
                    "Legacy-Sampler",
                )
                @test_throws ArgumentError MCBench.build_teststatistic(
                    legacy_testcase,
                    [count_metric];
                    s=legacy_sampler,
                    n=1,
                    n_steps=7,
                    use_sampler=false,
                    par=false,
                    clean=false,
                )
                @test legacy_sampler.draw_count == 0
            end
        end
    end
end
