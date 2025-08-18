"""
Manually start a vLLM server.
"""

import asyncio
import logging
import sys

from flag_safety.server.vllm_server import VLLMServer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Example usage of the VLLMServer class."""
    # Example configuration
    server = VLLMServer(
        model_path="/share/project/juntao/Projects/oss-red-team/models/openai/gpt-oss-20b",
        model_name="gpt-oss-20b",
        port=8001,
        host="0.0.0.0",
        pipeline_parallel_size=1,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9,
        limit_mm_per_prompt=None,
        chat_template=None,
        enable_prefix_caching=True,
        max_seq_len=131072,
        dtype="bfloat16",
        disable_log_stats=True,
        disable_log_requests=True,
        disable_fastapi_docs=True,
        uvicorn_log_level="warning",
        server_init_timeout=300,
    )

    try:
        api_base = await server.start()
        served_name_list = server.args.get("served_model_name")
        logger.info(f"Server started at {api_base} with model {served_name_list}")

        # Keep server running for a while until the user enter "exit" or subprocess is killed
        print('\033[93mType "q" to stop the server, or press Ctrl+C\033[0m')
        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    await asyncio.sleep(1)
                    continue
                if line.strip().lower() in {"exit", "quit", "q"}:
                    logger.info("Exit command received. Shutting down server...")
                    break
            except (KeyboardInterrupt, EOFError):
                logger.info("Interrupt received. Shutting down server...")
                break

    finally:
        await server.stop()

if __name__ == "__main__":
    asyncio.run(main())
