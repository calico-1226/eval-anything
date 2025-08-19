import logging

from flag_safety.benchmarks import BenchmarkRegistry
from flag_safety.client import ResponseClient, InferenceConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

base_url = "http://localhost:8001/v1"
api_key = "EMPTY"
model_name = "gpt-oss-20b"
reasoning_effort = "high"

model_client = ResponseClient(
    base_url=base_url,
    api_key=api_key,
    inference_config=InferenceConfig(
        model_name=model_name,
        reasoning_effort=reasoning_effort,
    ),
)

# Create the evaluator
evaluator_kwargs = {
    "results_dir": "./results",
}

benchmark = BenchmarkRegistry.create("DoNotAnswer", **evaluator_kwargs)

# Run the evaluation
logger.info("Running evaluation on TruthfulQA benchmark")
results = benchmark.evaluate(
    model_client=model_client,
)
print(results)
