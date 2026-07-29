from milabench.pack import Package


class BraxBenchmark(Package):
    base_requirements = "requirements.in"
    main_script = "main.py"

    def make_env(self):
        env = super().make_env()
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
        env["BENCHMATE_TORCHMEM"] = "0"
        env["BENCHMATE_JAXMEM"] = "1"
        return env
    
__pack__ = BraxBenchmark
