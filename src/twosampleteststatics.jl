"""Calculate a two-sample metric from two already prepared sample vectors."""
function run_teststatistic(
    testcase::AbstractTestcase,
    first::DensitySampleVector,
    second::DensitySampleVector,
    metric::TwoSampleMetric,
    ::AnySampler,
)
    calc_metric(testcase, first, second, metric)
end

"""Compare provided samples with a newly drawn IID reference sample."""
function run_teststatistic(
    testcase::AbstractTestcase,
    samples::DensitySampleVector,
    metric::TwoSampleMetric,
    ::AnySampler,
)
    iid_samples = sample(testcase; n_steps=length(samples))
    calc_metric(testcase, iid_samples, samples, metric)
end

function run_teststatistic_two_sample_metric(
    testcase::AbstractTestcase,
    samples::DensitySampleVector,
    metric::TwoSampleMetric,
    ::Int,
)
    iid_samples = sample(testcase; n_steps=length(samples))
    calc_metric(testcase, iid_samples, samples, metric)
end

"""Draw both IID and algorithm samples before calculating a two-sample metric."""
function run_teststatistic(
    testcase::AbstractTestcase,
    metric::TwoSampleMetric,
    sampler::AnySampler;
    n_steps::Int=100_000,
)
    iid_samples = sample(testcase; n_steps=n_steps)
    sampler_samples = sample(testcase, sampler; n_steps=n_steps)
    calc_metric(testcase, iid_samples, sampler_samples, metric)
end

"""Calculate the IID baseline distribution of a two-sample metric."""
function run_teststatistic(
    testcase::AbstractTestcase,
    metric::TwoSampleMetric;
    n_steps::Int=100_000,
)
    first = sample(testcase; n_steps=n_steps)
    second = sample(testcase; n_steps=n_steps)
    calc_metric(testcase, first, second, metric)
end
