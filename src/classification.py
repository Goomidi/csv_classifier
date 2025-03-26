import asyncio
import json
from collections.abc import Callable

import pandas as pd
import streamlit as st
from openai import AsyncOpenAI

from src.core.config import settings
from src.schemas.analysis_schema import AnalysisSchema


class TextClassifier:
    def __init__(self, model: str = "openai/gpt-4o-mini"):
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY, base_url=settings.LITE_LLM_BASE_URL
        )
        self.model = model
        self.categories = []
        self.category_descriptions = {}

    def set_categories(self, categories_dict: dict[str, str]) -> None:
        """Set the classification categories with descriptions.

        Args:
            categories (dict[str, str]): A dictionary containing category names and descriptions.
        """
        self.categories = list(categories_dict.keys())
        self.category_descriptions = categories_dict

    async def classify_text(self, text: str) -> AnalysisSchema:
        """Classify a single text input using LLM asynchronously.

        Args:
            text (str): The text to classify.
            with_explanation (bool): Whether to include an explanation in the response.

        Returns:
            OpenAISchema: A dictionary containing the classification result.
        """
        if not text or pd.isna(text):
            return AnalysisSchema(
                category="Empty",
                confidence=1.0,
                keywords=[],
                explanation="Empty or missing text",
                ambiguities=[],
            )

        if not self.categories:
            raise ValueError("Categories must be set before classification")

        categories_info = "\n".join(
            [f"- {cat}: {self.category_descriptions[cat]}" for cat in self.categories]
        )

        categories_info += "\n- Undetermined: The text doesn't clearly fit any of the defined categories."

        system_message = f"""You are a text classification system.
        Classify the provided text into ONE of the following categories:
        {categories_info}

        If a text could fit multiple categories, select the MOST appropriate one.

        Respond in JSON format with these fields:
        - "category" (string): The selected category name
        - "confidence" (float): A number between 0 and 1 indicating your confidence
        - "explanation" (string): A brief explanation of why this category was chosen
        - "keywords" (list[string]): A list of keywords explaining your decision
        - "ambiguities" (list[dict[string, string]]): If the text seems to fit multiple categories, list them here. It should be a list of objects, each containing a category name and an explanation.
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Text to classify: {text}"},
                ],
                temperature=0.1,
            )

            result = AnalysisSchema(**json.loads(response.choices[0].message.content))
            return result

        except Exception as e:
            error_msg = f"Classification error: {e!s}"
            st.error(error_msg)
            return AnalysisSchema(
                category="Error",
                confidence=0,
                keywords=[],
                explanation=f"Error during classification: {e!s}",
                ambiguities=[],
            )

    async def batch_classify(
        self, texts: list[str], progress_callback: Callable | None = None
    ) -> list[AnalysisSchema]:
        """Classify a batch of texts asynchronously with optional progress callback.

        Args:
            texts (list[str]): The texts to classify.
            progress_callback (callable, optional): A callback function to report progress.

        Returns:
            list[OpenAISchema]: A list of OpenAISchema containing the classification results.
        """
        total = len(texts)
        completed = 0

        async def process_with_progress(
            text: str, index: int
        ) -> tuple[int, AnalysisSchema]:
            result = await self.classify_text(text)

            if progress_callback:
                nonlocal completed
                completed += 1
                progress_callback(completed / total)

            return index, result

        tasks = [process_with_progress(text, i) for i, text in enumerate(texts)]

        semaphore = asyncio.Semaphore(5)

        async def bounded_process(
            task: tuple[int, AnalysisSchema],
        ) -> tuple[int, AnalysisSchema]:
            async with semaphore:
                return await task

        results_with_indices = await asyncio.gather(
            *[bounded_process(task) for task in tasks]
        )

        sorted_results = [
            r for _, r in sorted(results_with_indices, key=lambda x: x[0])
        ]

        return sorted_results
