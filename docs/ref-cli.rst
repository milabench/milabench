
milabench.cli
=============

The ``cli`` module provides all command-line entry points for milabench.
The main entry point is ``milabench <subcommand>``.

Core Commands
-------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Command
     - Description
   * - ``milabench install``
     - Install benchmark dependencies into isolated virtual environments.
   * - ``milabench prepare``
     - Download datasets, model weights, and other required data.
   * - ``milabench run``
     - Run benchmarks and collect metrics.
   * - ``milabench report``
     - Generate a report aggregating all runs into a final summary.
   * - ``milabench prepare_run``
     - Prepare and run in a single step.

Reporting & Analysis
--------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Command
     - Description
   * - ``milabench summary``
     - Produce a JSON summary of a previous run.
   * - ``milabench compare``
     - Compare all runs with each other.
   * - ``milabench publish``
     - Publish an archived run to a database.

Cluster & Infrastructure
------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Command
     - Description
   * - ``milabench slurm_system``
     - Generate a ``system.yaml`` from Slurm environment variables.
   * - ``milabench schedule``
     - Launch a Slurm job to run milabench.
   * - ``milabench machine``
     - Display machine metadata (GPU info, hostname, etc.).
   * - ``milabench container``
     - Run milabench inside a container.
   * - ``milabench multirun``
     - Run milabench multiple times with different configurations.
   * - ``milabench tunnel``
     - Set up port forwarding for multi-node communication.

Development & Utilities
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Command
     - Description
   * - ``milabench new``
     - Create a new benchmark from a template.
   * - ``milabench dev``
     - Open a shell in a benchmark's environment for development.
   * - ``milabench pin``
     - Pin benchmark dependencies.
   * - ``milabench pip``
     - Run pip across every benchmark pack.
   * - ``milabench resolve``
     - Resolve and display the final merged configuration.
   * - ``milabench env``
     - Print milabench environment variables.
   * - ``milabench gated``
     - Check access to gated Hugging Face models.
   * - ``milabench sharedsetup``
     - Restore data from a shared/network location to local disk.
   * - ``milabench archive``
     - Create deterministic tar archives of data and cache for sharing.
   * - ``milabench prefer_system``
     - Uninstall local packages that shadow system-provided ones.
   * - ``milabench ci``
     - Output benchmark groups as JSON for CI matrix generation.
   * - ``milabench patch``
     - Apply global patches to benchmark environments.
   * - ``milabench replay``
     - Replay a previously recorded run.


Module Reference
----------------

.. automodule:: milabench.cli
    :members: Main
    :undoc-members:
    :show-inheritance:
