# Benchmark Integration Guide

This document guides developers on how to integrate new benchmarks into this evaluation framework.

## Integration Overview

To integrate a new benchmark, you need to subclass the abstract base class [`BenchmarkEvaluator`](flag_safety/benchmarks/evaluation.py) and implement all of its abstract methods. The main steps are:

1.  Create a new benchmark directory
2.  Implement the evaluator class (subclassing `BenchmarkEvaluator`)
3.  Register the new benchmark
4.  Provide the necessary dataset and utility functions

## Example Directory Structure

```plaintext
benchmarks/
├── __init__.py
├── registry.py
└── my_benchmark/
    ├── __init__.py
    ├── eval.py        # Contains the evaluator implementation
    └── utils.py       # Helper utility functions (usually adapted from the benchmark's official GitHub repository)
```

## Abstract Method Descriptions

All abstract methods in the `BenchmarkEvaluator` class must be implemented in your subclass. Below are detailed descriptions for each method:

### `load_dataset`

```python
@abstractmethod
def load_dataset(self, split: str = "test") -> Any:
    """Load the benchmark dataset."""
    pass
```

**Purpose**: Loads and preprocesses the dataset required for the benchmark.

### `prepare_input_item`

```python
@abstractmethod
def prepare_input_item(self, item: Dict[str, Any]) -> Tuple[str, str, Any]:
    """Prepare the input for each data item."""
    pass
```

**Purpose**: Processes a single data instance from the dataset and organizes it into dataclass `EvaluationItem`.

### `evaluate_item`

```python
@abstractmethod
def evaluate_item(self, item: Dict[str, Any], response: str) -> Dict[str, Any]:
    """Evaluate the model's output for a single data item."""
    pass
```

**Purpose**: Assesses the model’s output on a single item to determine correctness or other evaluation metrics.

## Optional Methods to Override

In addition to the required abstract methods, you can optionally override the following methods to add custom functionality:

### 1. `calculate_metrics`

```python
def calculate_metrics(self, detailed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate metrics from detailed results."""
```

**Purpose**: Summarizes the results of all evaluated items and computes overall metrics (e.g., accuracy, ASR).

## Registering a New Benchmark

Use the `BenchmarkRegistry.register` decorator to register your new benchmark:

```python
from benchmarks.registry import BenchmarkRegistry

@BenchmarkRegistry.register("my_benchmark")
class MyBenchmarkEvaluator(BenchmarkEvaluator):
    # Implement abstract methods
    ...
```

Also, you need to import the new evaluator class in `benchmarks/__init__.py` so the framework can automatically discover and register it:

```python
# __init__.py
from benchmarks.my_benchmark.eval import MyBenchmarkEvaluator
```

## Full Implementation Example

For a complete implementation example, please refer to [latent_jailbreak](flag_safety/benchmarks/latent_jailbreak) for T2T and [mmmu](flag_safety/benchmarks/mmmu) for TI2T.

## Testing the New Benchmark

Once integration is complete, you can run the scripts in the `scripts` folder to test your benchmark. Refer [latent_jailbreak script](local_scripts/latent_jailbreak.sh) for T2T and [mmmu script](local_scripts/mmmu.sh) for TI2T.

## General Principles and Important Notes

When adding a new benchmark, please adhere to the following principles and considerations:

1.  **Benchmark Type**: Currently, it is recommended to prioritize the integration of rule-based benchmarks. API-based and local model-based evaluations may still have potential bugs and should be approached with caution.
2.  **Official Evaluation Code**: If official evaluation code for the benchmark exists, strive to use it as closely as possible. Any helper functions or utilities derived from official sources should be placed in the `utils.py` file within your benchmark's directory, with clear references to the original source (e.g., a comment indicating `ref: [link to official code/paper]`).
3.  **Development Workflow**:
    *   Always fork the main repository to your personal account or organization.
    *   Create a new branch in your forked repository for developing the new benchmark.
    *   Do **not** create branches directly in the main repository for new benchmark development.
    *   Once your benchmark integration is complete and tested, submit a Pull Request (PR) from your forked repository's branch to the main repository.
4.  **Pull Request Requirements**: When submitting a PR for a new benchmark, ensure that you have run at least one model through your benchmark and include a brief summary of the results or any notable observations in your PR description. This helps in verifying the basic functionality and correctness of the integration.
5.  **Correctness**: Aim to keep the evaluation process consistent with the official implementation. If the official evaluation code is not open-sourced, refer to other open-source evaluation frameworks for guidance.
6.  **Ray-Related Issues**: This framework uses Ray for parallel execution. If you encounter hanging processes, open a new terminal and run `ray stop`.
7.  **Datasets**:
    * If the dataset is officially available on HuggingFace, use the `hf_link` to load it directly instead of integrating it into the framework.
    * If there is no official dataset on HuggingFace, please temporarily convert the data into the HuggingFace dataset format and upload it to your personal HuggingFace repository.
8.  **Code Development within Subfolders**: Always develop benchmark code by subclassing and overriding methods within the benchmark’s subfolder. If you need to modify other parts of the framework, please discuss it in advance with the project maintainers.
