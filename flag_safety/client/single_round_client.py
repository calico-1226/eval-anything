from __future__ import annotations

import logging
from typing import Any

from flag_safety.utils.cached_requests import cached_requests
from flag_safety.utils.data_type import EvaluationItem
from flag_safety.utils.parallel_processing import parallel_processing_backend

# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SingleRoundClient:
    """Client for single round dialog."""

    def __init__(
        self,
        api_base: str,
        model: str,
        api_key: str = "EMPTY",
        cache_dir: str = "./cache",
        num_workers: int = 100,
        **kwargs,
    ):
        """
        Initialize the single round dialog client.

        Args:
            api_base: Base URL for the API
            api_key: API key for the API
            model: Name of the model to use
            cache_dir: Directory to cache results
            num_workers: Maximum number of parallel workers
        """
        self.api_base = api_base
        self.model = model
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.num_workers = num_workers

    def _inference_item_wrapper(self, param: dict[str, Any]) -> EvaluationItem:
        """Wrapper for inference_item for parallel processing."""
        return self.inference_item(
            evaluation_item=param["evaluation_item"],
            temperature=param["temperature"],
            top_p=param["top_p"],
            repetition_penalty=param["repetition_penalty"],
            max_tokens=param["max_tokens"],
        )

    def inference_item(
        self,
        evaluation_item: EvaluationItem,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        max_tokens: int,
    ) -> EvaluationItem:
        """Calls the cached request function for a single inference."""
        response = cached_requests(
            messages=evaluation_item.conversation,
            model=self.model,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_completion_tokens=max_tokens,
            api_key=self.api_key,
            api_base=self.api_base,
            cache_dir=self.cache_dir,
        )
        evaluation_item.add_text_content(response)
        return evaluation_item

    def run(
        self,
        evaluation_items: list[EvaluationItem],
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        max_tokens: int,
    ) -> list[str]:
        """
        Run inference on the model using parallel processing.

        Args:
            messages_list: List of message lists
            temperature: Temperature parameter for sampling
            top_p: Top-p parameter for sampling
            repetition_penalty: Repetition penalty parameter
            max_tokens: Maximum tokens for the response

        Returns:
            List of responses
        """
        params = [
            {
                "evaluation_item": evaluation_item,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "max_tokens": max_tokens,
            }
            for evaluation_item in evaluation_items
        ]

        evaluation_items = parallel_processing_backend(
            params=params,
            fn=self._inference_item_wrapper,
            num_workers=self.num_workers,
            desc="Inferring",
        )
        return evaluation_items
