@testset "Sampler implementations" begin
    @testset "plain file sampler" begin
        mktempdir() do dir
            first_path = write_lines(joinpath(dir, "01.txt"), ["1,2", "3,4"])
            second_path = write_lines(joinpath(dir, "02.txt"), ["5,6"])
            sampler = MCBench.FileBasedSampler([first_path, second_path]; info="Plain-Files")

            try
                @test sampler.info == "Plain-Files"
                @test MCBench.read_sample!(sampler) == "1,2"
                @test MCBench.read_sample!(sampler) == "3,4"
                @test MCBench.read_sample!(sampler) == "5,6"
                @test_throws ErrorException MCBench.read_sample!(sampler)

                MCBench.reset_sampler!(sampler)
                @test MCBench.read_sample!(sampler) == "1,2"

                MCBench.reset_sampler!(sampler)
                samples = MCBench.sample(sampler; n_steps=2)
                @test sample_matrix(samples) == [1.0 3.0; 2.0 4.0]
            finally
                MCBench.close_sampler!(sampler)
            end

            directory_sampler = MCBench.FileBasedSampler(dir)
            try
                @test directory_sampler.files == sort([first_path, second_path])
            finally
                MCBench.close_sampler!(directory_sampler)
            end

            @test_throws ArgumentError MCBench.FileBasedSampler(String[])
            @test_throws ArgumentError MCBench.FileBasedSampler(joinpath(dir, "missing.txt"))
        end
    end

    @testset "CSV sampler, headers, and masks" begin
        mktempdir() do dir
            first_path = write_lines(joinpath(dir, "01.csv"), ["a,b,c", "1,10,100", "2,20,200"])
            second_path = write_lines(joinpath(dir, "02.csv"), ["a,b,c", "3,30,300"])
            sampler = MCBench.CsvBasedSampler([first_path, second_path]; info="CSV-Files")

            try
                @test sampler.header == ["a", "b", "c"]
                @test MCBench.read_sample!(sampler) == "1,10,100"
                @test MCBench.read_sample!(sampler) == "2,20,200"
                @test MCBench.read_sample!(sampler) == "3,30,300"

                MCBench.reset_sampler!(sampler)
                MCBench.set_mask(sampler, ["c", "a"])
                samples = MCBench.sample(sampler, 2)
                @test sample_matrix(samples) == [100.0 200.0; 1.0 2.0]

                MCBench.reset_sampler!(sampler)
                testcase = standard_normal_testcase(2)
                MCBench.set_mask(sampler, ["a", "b"])
                testcase_samples = MCBench.sample(testcase, sampler; n_steps=2)
                @test sample_matrix(testcase_samples) == [1.0 2.0; 10.0 20.0]
                @test testcase_samples.logd ≈ logpdf(testcase.f, sample_matrix(testcase_samples))
            finally
                MCBench.close_sampler!(sampler.fbs)
            end

            mismatched_path = write_lines(
                joinpath(dir, "03.csv"),
                ["a,d,c", "4,40,400"],
            )
            mismatched = MCBench.CsvBasedSampler(
                [first_path, mismatched_path],
            )
            try
                MCBench.read_sample!(mismatched)
                MCBench.read_sample!(mismatched)
                @test_throws ArgumentError MCBench.read_sample!(mismatched)
                @test_throws ArgumentError MCBench.set_mask(mismatched, ["missing"])
            finally
                MCBench.close_sampler!(mismatched)
            end
        end
    end

    @testset "density-sample sampler" begin
        first_sample = MCBench.make_dsv([1.0, 2.0])
        second_sample = MCBench.make_dsv([3.0])
        sampler = MCBench.DsvSampler([first_sample, second_sample]; info="DSV-Files")

        @test sampler.total_samples == 3
        @test !sampler.weighted
        @test MCBench.read_sample!(sampler).v == first_sample[1].v
        @test MCBench.read_sample!(sampler).v == first_sample[2].v
        @test MCBench.read_sample!(sampler).v == second_sample[1].v
        @test_throws ErrorException MCBench.read_sample!(sampler)

        sampler = MCBench.DsvSampler([first_sample]; info="DSV-Resampling")
        Random.seed!(33)
        @test length(MCBench.sample(sampler; n_steps=2)) == 2
        @test_throws ArgumentError MCBench.DsvSampler(typeof(first_sample)[])

        weighted = MCBench.make_dsv(
            [1.0, 2.0, 3.0];
            weights=[1.0, 2.0, 1.0],
        )
        weighted_sampler = MCBench.DsvSampler([weighted])
        MCBench.unweight!(weighted_sampler)
        @test !weighted_sampler.weighted
        @test !MCBench.is_weighted(only(weighted_sampler.dsvs))
    end

    @testset "BAT Metropolis-Hastings configuration" begin
        sampler = MCBench.BATMH(n_steps=20, nchains=1)
        @test sampler isa MCBench.MCMCSamplingAlgorithm
        @test sampler.info == "BAT-MH"
        @test sampler.sampler isa BAT.AbstractSamplingAlgorithm
        @test MCBench.buildBATPosterior(standard_normal_testcase(1)) isa BAT.PosteriorMeasure
    end
end
