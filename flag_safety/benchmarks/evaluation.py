import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from flag_safety.utils.data_type import EvaluationItem
from flag_safety.utils.parallel_processing import parallel_processing_backend

# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BenchmarkEvaluator(ABC):
    """Abstract base class for benchmark evaluators."""

    def __init__(
        self,
        data_path: str = None,
        model_name: Optional[str] = None,
        results_dir: str = "./results",
        benchmark: str = "benchmark",
        **kwargs,
    ):
        """
        Initialize a benchmark evaluator.

        Args:
            model_name: Name of the model being evaluated
            results_dir: Directory to save results
        """
        self.model_name = model_name
        self.data_path = data_path
        self.benchmark = benchmark
        if self.data_path is None:
            raise ValueError("data_path must be provided")
        self.results_dir_base = os.path.join(results_dir, self.benchmark, model_name)
        os.makedirs(self.results_dir_base, exist_ok=True)

    @abstractmethod
    def load_dataset(self, split: str = "test") -> Any:
        """
        Load the benchmark dataset.

        Args:
            split: Dataset split to load (e.g., "train", "test", "validation")

        Returns:
            The loaded dataset
        """
        raise NotImplementedError(
            "Method load_dataset should be implemented by the subclass"
        )

    @abstractmethod
    def prepare_input_item(self, item: Dict[str, Any]) -> EvaluationItem:
        """
        Prepare input for a single item from the dataset.

        Args:
            item: A single item from the dataset

        Returns:
            Tuple of (system_content, user_content, image)
        """
        raise NotImplementedError(
            "Method prepare_input_item should be implemented by the subclass"
        )

    @abstractmethod
    def evaluate_item(
        self, item: Dict[str, Any], evaluation_item: EvaluationItem
    ) -> Dict[str, Any]:
        """
        Evaluate a single model output against an item from the dataset.

        Args:
            item: A single item from the dataset
            response: Model's response for this item

        Returns:
            Evaluation results for this item
        """
        raise NotImplementedError(
            "Method evaluate_item should be implemented by the subclass"
        )

    def _prepare_input_item_wrapper(self, param: Dict[str, Any]) -> EvaluationItem:
        """
        Wrapper function for prepare_input_item to use with parallel processing.

        Args:
            param: Dictionary containing the item

        Returns:
            EvaluationItem
        """
        return self.prepare_input_item(param["item"])

    def prepare_inputs(
        self, dataset: Any, num_workers: int = 100
    ) -> List[EvaluationItem]:
        """
        Prepare inputs for the budget forcing client using parallel processing.

        Args:
            dataset: The loaded dataset
            num_workers: Number of parallel workers

        Returns:
            Tuple of (system_contents, user_contents, images)
        """
        # Prepare parameter list for parallel processing
        params = [{"item": item} for item in dataset]

        # Run parallel processing
        logger.info(
            f"Preparing inputs for {len(params)} examples using {num_workers} workers"
        )
        return parallel_processing_backend(
            params=params,
            fn=self._prepare_input_item_wrapper,
            num_workers=num_workers,
            desc="Preparing Inputs",
        )

    def _evaluate_item_wrapper(self, param: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wrapper function for evaluate_item to use with parallel processing.

        Args:
            param: Dictionary containing the item and response

        Returns:
            Evaluation results for this item
        """
        return self.evaluate_item(param["item"], param["evaluation_item"])

    def evaluate_outputs(
        self,
        dataset: Any,
        evaluation_items: List[EvaluationItem],
        num_workers: int = 100,
    ) -> Dict[str, Any]:
        """
        Evaluate model outputs against the benchmark using parallel processing.

        Args:
            dataset: The loaded dataset
            evaluation_items: Model outputs from inference
            num_workers: Number of parallel workers

        Returns:
            Dictionary with evaluation results
        """
        assert len(evaluation_items) == len(dataset)

        # Prepare parameter list for parallel processing
        params = [
            {"item": item, "evaluation_item": evaluation_item}
            for item, evaluation_item in zip(dataset, evaluation_items)
        ]

        # Run parallel processing
        logger.info(f"Evaluating {len(params)} examples using {num_workers} workers")
        detailed_results = parallel_processing_backend(
            params=params,
            fn=self._evaluate_item_wrapper,
            num_workers=num_workers,
            desc="Evaluating Outputs",
        )

        # Calculate metrics using the potentially overridden method
        return self.calculate_metrics(detailed_results)

    def calculate_metrics(
        self, detailed_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate metrics based on detailed evaluation results.

        This method can be overridden by derived classes to implement custom metrics.

        Args:
            detailed_results: List of detailed evaluation results for each item

        Returns:
            Dictionary with evaluation metrics
        """
        num_total = len(detailed_results)

        # Count correct answers
        num_match = sum(
            1 for result in detailed_results if result.get("correct", False)
        )
        accuracy = num_match / num_total if num_total > 0 else 0

        return {
            "accuracy": accuracy,
            "num_correct": num_match,
            "num_total": num_total,
            "detailed_results": detailed_results,
        }

    def run_evaluation(
        self,
        client,
        temperature: float = 0.3,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        max_tokens: int = 32000,
        split: str = "test",
        num_workers: int = 100,
    ) -> Dict[str, Any]:
        """
        Run an evaluation on the benchmark.

        Args:
            client: Client instance
            temperature: Temperature parameter for sampling
            top_p: Top-p parameter for sampling
            repetition_penalty: Repetition penalty parameter
            max_tokens: Maximum tokens for inference
            split: Dataset split to evaluate on
            num_workers: Number of parallel workers for input preparation and evaluation

        Returns:
            Dictionary with evaluation results
        """
        # Load dataset
        logger.info(f"Loading dataset for {self.benchmark} split '{split}'")
        dataset = self.load_dataset(split)

        # Prepare inputs using parallel processing
        evaluation_items = self.prepare_inputs(dataset, num_workers=num_workers)

        # Run inference
        logger.info(f"Running inference on {len(evaluation_items)} examples")
        evaluation_items = client.run(
            evaluation_items=evaluation_items,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
        )

        # Evaluate outputs for each num_ignore value using parallel processing
        logger.info(f"Evaluating {len(evaluation_items)} outputs")
        eval_results = self.evaluate_outputs(
            dataset, evaluation_items, num_workers=num_workers
        )
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        current_results_dir = os.path.join(
            self.results_dir_base,
            (
                f"{timestamp}_temp_{str(temperature).replace('.', '_')}"
                f"_top_p_{str(top_p).replace('.', '_')}"
                f"_rep_penalty_{str(repetition_penalty).replace('.', '_')}"
            ),
        )
        os.makedirs(current_results_dir, exist_ok=True)

        result_file = os.path.join(current_results_dir, "result.json")

        # Consolidate results into a single dictionary
        full_results = {
            "hyperparameters": {
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "max_tokens": max_tokens,
                "num_workers": num_workers,
            },
            "timestamp": timestamp,
            "benchmark": self.benchmark,
            "model": self.model_name,
            "split": split,
            **eval_results,
        }

        logger.info(f"Saving full results to {result_file}")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=4, ensure_ascii=False)

        logger.info(
            f"Evaluation complete for {self.benchmark} on model {self.model_name}"
        )
        logger.info(f"Accuracy: {eval_results.get('accuracy', 'N/A')}")

        return full_results
