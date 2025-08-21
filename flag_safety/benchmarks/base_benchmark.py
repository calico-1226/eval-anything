from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from flag_safety.client.response_client import ResponseClient
import os
import json

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

        # initialize logs
        self.logs = None

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

    def save_logs(self) -> None:
        assert "inference_config" in self.logs, "inference_config is not in logs"
        save_dir = os.path.join(
            self.results_dir,
            self.BENCHMARK_NAME,
            self.logs["inference_config"]["model_name"],
        )
        os.makedirs(save_dir, exist_ok=True)
        for key, value in self.logs.items():
            save_path = os.path.join(save_dir, f"{key}.json")
            with open(save_path, "w") as f:
                json.dump(value, f, indent=4, ensure_ascii=False)
        logger.info(f"Logs saved to {save_dir}")

    def evaluate(self, model_client: ResponseClient) -> dict:

        self.logs = {"inference_config": model_client.inference_config.to_dict()}

        responses = model_client.parallel_get_responses(self.messages)
        metrics = self.calculate_metrics(responses)

        self.save_logs()
        return metrics
