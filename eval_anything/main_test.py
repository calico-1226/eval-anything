import logging

from eval_anything.benchmarks import BenchmarkRegistry
from eval_anything.client import ResponseClient, InferenceConfig

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
        # temperature=0.01,
        reasoning_effort=reasoning_effort,
    ),
    enable_cache=True,
    cache_dir="./.cache",
)

# Create the evaluator
evaluator_kwargs = {
    "results_dir": "./results",
    "judger_model": "gpt-5-mini",
}

benchmark = BenchmarkRegistry.create("BeaverTails", **evaluator_kwargs)

# Run the evaluation
results = benchmark.evaluate(
    model_client=model_client,
)
print(results)
