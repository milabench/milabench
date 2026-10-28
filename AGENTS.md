


Code
----

* voir: is used for metrics collection (gpu, cpu, network)
    * `voir.instruments.*` implements the metric collection for the system
    * DO NOT implement it elsewhere

* benchmate: is home to common utilities reused by benchmarks

* Prefer using `benchmate.metrics.TimedIterator` to record performance as the unified way.
    * Fallback to `benchmate.metrics.ManualTimedIterator`
        * Then Fallaback to `benchmate.timings.StepTimer`
            * Only fallback to a custom instrumentation if the benchmark is illsuited for the instrumented
            iterator (RL, Simulation, etc...)

* Route metrics to give/voir logging for rank0
* Other rank metrics should go to the rankstore

Layout
------

Milabench should be able to run offline once `milabench install` and `milabench prepare` have been run.

`$MILABENCH_BASE/data` should hold all the necessary data for `milabench run` to work.

`$MILABENCH_BASE/cache` is only there to avoid 3rd party filling up the wrong location (`~/.cache`) that might not be big enough to hold the data.

All metrics should be gathered in `$MILABENCH_BASE/runs/<runname>/*.data`


Dependencies
------------

All dependencies should be defined in `requirements.toml` inside the benchmark directory.
it has a `common` section and accelerator specific section for accelerator specific dependencies.

Dependencies gets pinned for a set of pytorch and backend version in `.pin`.
Benchmark are installed by generating a requirement file from the `requirements.toml` using the right constraint file in `.pin`

for example `milabench install llama --set cuda=132 torch=2.13` 


    requirements.txt = generate_requirements(llama/requirements.toml, 'cuda')

    uv pip install -r requirements.txt -c .pin/constraints.cuda132.torch2130.txt


Complex dependencies should use `benchfile.py:Package.install` to implement the install process.
