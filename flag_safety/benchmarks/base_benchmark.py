from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from flag_safety.client.response_client import ResponseClient

# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BaseBenchmark(ABC):
    """Abstract base class for benchmark evaluators."""

    BENCHMARK_NAME = "UNDEFINED"

    def __init__(
        self,
        results_dir: str = "./results",
        **kwargs,
    ):
        # set parameters
        self.results_dir = results_dir

        # load dataset
        self.raw_dataset = self.load_dataset()

        # prepare messages
        self.messages = self.prepare_messages(self.raw_dataset)

    @abstractmethod
    def load_dataset(self, *args, **kwargs) -> Any:
        raise NotImplementedError(
            "Method load_dataset should be implemented by the subclass"
        )

    @abstractmethod
    def prepare_messages(self, dataset: Any) -> list[list[dict[str, Any]]]:
        raise NotImplementedError(
            "Method prepare_messages should be implemented by the subclass"
        )

    @abstractmethod
    def calculate_metrics(self, responses: list[str]) -> dict:
        raise NotImplementedError(
            "Method calculate_metrics should be implemented by the subclass"
        )

    def evaluate(self, model_client: ResponseClient) -> dict:
        responses = model_client.parallel_get_responses(self.messages)
        return self.calculate_metrics(responses)
