
Overview
========

Milabench is a benchmarking suite for GPU-accelerated machine learning workloads.
It covers a wide range of models and training scenarios, from single-GPU to multi-node distributed training.

See the :doc:`quickstart` page for full installation and usage instructions.

.. include:: ../_gpu_summary.rst


Prerequisites
-------------

Hugging Face Access
^^^^^^^^^^^^^^^^^^^

Several benchmarks use gated models that require explicit access approval on Hugging Face.

1. Request access to the following models:

   - `Llama-2-7b <https://huggingface.co/meta-llama/Llama-2-7b>`_ - llama inference
   - `Llama-3.1-8B <https://huggingface.co/meta-llama/Llama-3.1-8B>`_ - llm-lora training
   - `Llama-3.1-70B <https://huggingface.co/meta-llama/Llama-3.1-70B>`_ - llm-lora-mp / llm-full-mp training
   - `Meta-Llama-3-8B-Instruct <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>`_ - vllm inference
   - `Llama-3.1-8B-Instruct <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>`_ - llm-chat inference
   - `Llama-4-Scout-17B-16E <https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E>`_ - vllm-scout / sglang inference

2. Create a `read token <https://huggingface.co/settings/tokens/new?tokenType=read>`_

3. Set the token in your environment:

   .. code-block:: bash

      export HF_TOKEN=<your_huggingface_token>


CLI Reference
-------------

Milabench provides a single CLI with the following subcommands:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Command
     - Description
   * - ``milabench install``
     - Install benchmark dependencies into virtual environments
   * - ``milabench prepare``
     - Download datasets, model weights, and other required data
   * - ``milabench run``
     - Run the benchmarks
   * - ``milabench report``
     - Generate a report from benchmark results
   * - ``milabench slurm_system``
     - Auto-generate a ``system.yaml`` from a Slurm allocation

Common flags:

.. code-block:: text

   --config <path>     Configuration file (default: config/standard.yaml)
   --base <path>       Base directory for envs, data, and runs
   --system <path>     System configuration file (nodes, SSH, docker)
   --select <pattern>  Only include matching benchmarks
   --exclude <pattern> Exclude matching benchmarks
   --repeat <n>        Run the suite multiple times
