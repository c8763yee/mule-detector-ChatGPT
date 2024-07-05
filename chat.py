from base64 import b64encode as b64e
from textwrap import dedent
from typing import TypeAlias

import openai
from dotenv import load_dotenv
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from config import OpenAIConfig, VisionConfig

JSON_TEXT: TypeAlias = str
openai_config = OpenAIConfig()


class ResultModel(BaseModel):
    prob: float
    flagged: bool
    analysis: str


class MuleProbModel(BaseModel):
    result: list[ResultModel]


class ChatGPT:
    """
    A chatbot based on OpenAI's chat API
    if the chat history doesn't need to save, then use DUMMY_UUID as UUID
    """

    behavior = {
        "role": "system",
        "content": dedent(
            """
        You are a helpful assistant to help me with my tasks.
        please answer my questions with my language.
        """
        ),
    }
    load_dotenv()
    client = openai.OpenAI()

    def __init__(self):
        self.detect_malicious_content(self.behavior["content"])
        self._history = [self.behavior]

    @classmethod
    def detect_malicious_content(cls, prompt: str) -> bool:
        response = cls.client.moderations.create(input=prompt)
        result = response.results[0]

        return result.flagged or any(
            cate is True for cate in result.categories.model_dump().values()
        )

    def setup_behavior(self, behavior: dict) -> None:
        self.behavior = behavior
        self._history[0] = behavior

    @classmethod
    def from_system_prompt(cls, prompt: str) -> "ChatGPT":
        instance = cls()
        instance.setup_behavior({"role": "system", "content": prompt})
        return instance

    def _send_message(
        self,
        **kwargs,
    ) -> ChatCompletion:
        response = self.client.chat.completions.create(
            messages=self._history, **{**openai_config.model_dump(), **kwargs}
        )
        return response

    def ask(
        self, prompt: str | list[dict], **open_kwargs
    ) -> tuple[str, CompletionUsage]:
        if isinstance(prompt, str) and self.detect_malicious_content(prompt):
            raise ValueError("This Prompt contains malicious content")

        self._history.append({"role": "user", "content": prompt})
        response = self._send_message(**open_kwargs)
        return response.choices[0].message.content, response.usage

    def vision(self, text: str, image_url: str, **openai_kwargs) -> JSON_TEXT:
        """
        returns the response from the vision model
        Args:
            text: the prompt to the model
            image_text: the base64 encoded image
        """
        if self.detect_malicious_content(text):
            raise ValueError("This Prompt contains malicious content")

        vision_prompt = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_url}",
                    "detail": VisionConfig.detail,
                },
            },
            {"type": "text", "text": text},
        ]
        return self.ask(vision_prompt, model="gpt-4o", **openai_kwargs)

    def visions(
        self, text, image_base64_list, **openai_kwargs
    ) -> tuple[str, CompletionUsage]:
        """
        returns the response from the vision model
        Args:
            text: the prompt to the model
            image_text: the base64 encoded image
        """
        if self.detect_malicious_content(text):
            raise ValueError("This Prompt contains malicious content")

        vision_prompt = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}",
                    "detail": VisionConfig.detail,
                },
            }
            for image_base64 in image_base64_list
        ]
        vision_prompt.append({"type": "text", "text": text})
        return self.ask(vision_prompt, model="gpt-4o", **openai_kwargs)


class MuleDetector:
    def __init__(self, system_prompt: str):
        self.client = ChatGPT.from_system_prompt(system_prompt)

    def start(
        self, image_paths: str, text: str = "", **openai_kwargs
    ) -> tuple[MuleProbModel, CompletionUsage]:
        image_base64_list = []

        for image_path in image_paths:
            if image_path.name.endswith((".png", ".jpg", ".jpeg")) is False:
                raise ValueError("Invalid Image Format")

            with open(image_path, "rb") as image:
                image_binary = image.read()
                image_base64 = b64e(image_binary).decode("utf-8")
                image_base64_list.append(image_base64)

        response, token_usage = self.client.visions(
            text, image_base64_list, **openai_kwargs
        )
        return MuleProbModel.model_validate_json(response), token_usage
