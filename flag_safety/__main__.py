import argparse
import asyncio
import logging
import os
from collections import defaultdict

import ray

from flag_safety.benchmarks import BenchmarkRegistry
from flag_safety.client.single_round_client import SingleRoundClient

# from server import ServerRegistry
from flag_safety.server.vllm_server import VLLMServer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Budget Forcing Evaluation Framework")

    # Server configuration
    parser.add_argument(
        "--server-backend",
        type=str,
        default="vllm",
        choices=["vllm", "remote"],
        help="Server backend to use. If using remote, no local server will be started.",
    )
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model to serve")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name to serve the model as (defaults to model path)",
    )
    parser.add_argument("--port", type=int, default=8010, help="Port to serve the API on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind to")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism",
    )
    parser.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for pipeline parallelism",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.7,
        help="Fraction of GPU memory to use",
    )
    parser.add_argument(
        "--limit-mm-per-prompt",
        type=str,
        default=None,
        help="Limit on multimedia per prompt",
    )
    parser.add_argument(
        "--chat-template", type=str, default=None, help="Path to chat template file"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=32768,
        help="Maximum sequence length for the model",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="Whether to enable prefix caching",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type for model weights")
    parser.add_argument("--disable-log-stats", action="store_true", help="Disable log stats")
    parser.add_argument("--disable-log-requests", action="store_true", help="Disable log requests")
    parser.add_argument("--disable-fastapi-docs", action="store_true", help="Disable fastapi docs")
    parser.add_argument(
        "--uvicorn-log-level", type=str, default="warning", help="Uvicorn log level"
    )
    parser.add_argument(
        "--disable-frontend-multiprocessing",
        action="store_true",
        help="Disable frontend multiprocessing to prevent nested multiprocessing",
    )
    parser.add_argument(
        "--server-init-timeout",
        type=int,
        default=300,
        help="Timeout in seconds for server initialization (default: 300s = 5min)",
    )

    # Server operation mode
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="Base URL for the API when using --no-server",
    )
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API key for the API")

    # Evaluation configuration
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=list(BenchmarkRegistry.list().keys()),
        help="Benchmark to evaluate on",
    )
    parser.add_argument("--data-path", type=str, default=None, help="Path to the benchmark dataset")
    parser.add_argument(
        "--results-dir", type=str, default="./results", help="Directory to save results"
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature parameter for sampling",
    )
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p parameter for sampling")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        help="Repetition penalty parameter",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="./cache", help="Directory to cache results"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=100,
        help="Maximum number of parallel workers",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate per response",
    )

    return parser.parse_args()


def display_args(args):
    """
    Display arguments in a structured, organized way.

    This function automatically categorizes and displays all arguments,
    making it extensible for future argument additions or removals.

    Args:
        args: The parsed command-line arguments
    """
    # Convert args namespace to dictionary
    args_dict = vars(args)

    # Define categories and their prefixes/keywords for automatic categorization
    categories = {
        "Server Configuration": [
            "model",
            "port",
            "host",
            "tensor",
            "pipeline",
            "gpu",
            "limit",
            "chat",
            "seq",
            "prefix",
            "dtype",
            "log",
            "fastapi",
            "uvicorn",
            "frontend",
            "server",
        ],
        "Server Operation": ["api-base", "api-key"],
        "Evaluation": ["benchmark", "data", "results", "cache", "split", "num-workers"],
        "Inference Parameters": ["temperature", "top-p", "repetition", "max-tokens"],
    }

    # Create a mapping of argument to category
    arg_category_map = {}
    for arg in args_dict:
        arg_category = "Other"  # Default category
        for category, keywords in categories.items():
            if any(keyword in arg for keyword in keywords):
                arg_category = category
                break
        arg_category_map[arg] = arg_category

    # Group arguments by category
    categorized_args = defaultdict(dict)
    for arg, value in args_dict.items():
        category = arg_category_map[arg]
        categorized_args[category][arg] = value

    # Display arguments by category
    logger.info("=== CONFIGURATION PARAMETERS ===")
    for category, args in categorized_args.items():
        logger.info(f"\n[{category}]")
        for arg, value in args.items():
            # Format the value based on its type
            if isinstance(value, bool):
                value_str = str(value)
            elif value is None:
                value_str = "None"
            elif isinstance(value, str) and len(value) > 50:
                value_str = f"{value[:47]}..."
            else:
                value_str = str(value)

            # Add information if this is a required parameter
            logger.info(f"  {arg:<25}: {value_str}")
    logger.info("===============================")


async def main():
    """Main entry point."""
    args = parse_args()

    # Display arguments in a structured format
    display_args(args)

    # Create necessary directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # Start the server if needed
    server = None
    api_base = args.api_base

    model_name = "gpt-oss-20b"
    server = VLLMServer(
        model_path="/share/project/juntao/Projects/oss-red-team/models/openai/gpt-oss-20b",
        model_name="gpt-oss-20b",
        port=8000,
        host="0.0.0.0",
        pipeline_parallel_size=1,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.7,
        limit_mm_per_prompt=None,
        chat_template=None,
        enable_prefix_caching=True,
        max_seq_len=32768,
        dtype="bfloat16",
        disable_log_stats=True,
        disable_log_requests=True,
        disable_fastapi_docs=True,
        uvicorn_log_level="warning",
        server_init_timeout=300,
    )
    try:
        api_base = await server.start()
        logger.info(f"Server started at {api_base}")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        raise e

    client = SingleRoundClient(
        api_base=api_base,
        model=model_name,
        api_key=args.api_key,
        cache_dir=args.cache_dir,
        num_workers=args.num_workers,
    )

    try:

        # Create the evaluator
        evaluator_kwargs = {
            "model_name": model_name,
            "cache_dir": args.cache_dir,
            "results_dir": args.results_dir,
            "num_workers": args.num_workers,
            "benchmark": args.benchmark,
            "data_path": args.data_path,
        }

        evaluator = BenchmarkRegistry.create(args.benchmark, **evaluator_kwargs)

        # Run the evaluation
        logger.info(f"Running evaluation on {args.benchmark} benchmark")

        results = evaluator.run_evaluation(
            client=client,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            split=args.split,
            max_tokens=args.max_tokens,
        )

        logger.info(f"Evaluation complete. Results saved to {args.results_dir}")

    finally:
        if server:
            logger.info("Stopping server")
            await server.stop()

        if ray.is_initialized():
            logger.info("Shutting down Ray")
            ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
