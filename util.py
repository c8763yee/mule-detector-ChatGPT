import json
import os
from asyncio import run
from base64 import b64encode as b64e
from pathlib import Path
from textwrap import dedent
from typing import TypeAlias
from uuid import uuid4

import edge_tts
import openai
import requests
from dotenv import load_dotenv
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from vtt_to_srt.vtt_to_srt import ConvertFile

from config import OpenAIConfig, VisionConfig

JSON_TEXT: TypeAlias = str
openai_config = OpenAIConfig()


class LineNotifyRequest(BaseModel):
    message: str
    imageThumbnail: str | None = None
    imageFullsize: str | None = None
    stickerPackageId: int | None = None
    stickerId: int | None = None
    notificationDisabled: bool = False


async def text2voice_subtitle(
    text: str,
    voice: str,
    audio_filepath: Path,
    *,
    rate: str,
    volume: str,
    pitch: str,
    proxy: str | None = None,
    receive_timeout: int = 60,
) -> Path:
    communicator = edge_tts.Communicate(
        text,
        voice,
        rate=rate,
        volume=volume,
        pitch=pitch,
        proxy=proxy,
        receive_timeout=receive_timeout,
    )
    submaker = edge_tts.SubMaker()
    with audio_filepath.open("wb") as file:
        async for chunk in communicator.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                submaker.create_sub((chunk["offset"], chunk["duration"]), chunk["text"])

    subtitle_path = audio_filepath.parent / (audio_filepath.stem + ".vtt")
    with subtitle_path.open("w", encoding="utf-8") as file:
        file.write(submaker.generate_subs())

    ConvertFile(str(subtitle_path), "utf-8").convert()
    with (subtitle_path.parent / (subtitle_path.stem + ".srt")).open("r", encoding="utf-8") as file:
        srt_content = file.read()

    with (subtitle_path.parent / (subtitle_path.stem + ".srt")).open("w", encoding="utf-8") as file:
        file.write(srt_content[srt_content.find("1") :])
    return audio_filepath


async def text2voice(
    text: str,
    voice: str,
    filepath: Path,
    *,
    rate: str,
    volume: str,
    pitch: str,
    proxy: str | None = None,
    receive_timeout: int = 60,
) -> Path:
    """Convert text to voice using edge_tts

    Args:
        text (str): The text to convert to voice
        voice (str, optional): TTS voices. Defaults to settings.VOICE.
        rate (str, optional): voice speed (format: "(+|-)x%"). Defaults to -15%.
        volume (str, optional): voice volume (format: "(+|-)x%"). Defaults to +0%.
        pitch (str, optional): _description_. Defaults to +0Hz.
        proxy (str | None, optional): _description_. Defaults to None.
        filepath (str, optional): _description_. Defaults to "media/dummy.mp3".

    Returns:
        str: The filepath of the generated audio
    """
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        volume=volume,
        pitch=pitch,
        proxy=proxy,
        receive_timeout=receive_timeout,
    )
    await communicate.save(filepath)
    return filepath.absolute()


def tts(
    text: str,
    voice: str = "zh-TW-HsiaoChenNeural",
    rate: str = "+0%",
    volume: str = "+30%",
    pitch: str = "+0Hz",
    filepath: str = "warning.mp3",
) -> Path:
    filepath = Path(filepath)
    return run(
        text2voice(
            text,
            voice,
            rate=rate,
            volume=volume,
            pitch=pitch,
            filepath=filepath,
        )
    )


def tts_with_subtitle(
    text: str,
    voice: str = "zh-TW-HsiaoChenNeural",
    rate: str = "+0%",
    volume: str = "+30%",
    pitch: str = "+0Hz",
    filepath: str = "warning.mp3",
):
    filepath = Path(filepath)
    return run(
        text2voice_subtitle(
            text,
            voice,
            rate=rate,
            volume=volume,
            pitch=pitch,
            audio_filepath=filepath,
        )
    )


def line_notify(message: str, image_paths: list[str] | None = None):
    url = "https://notify-api.line.me/api/notify"
    headers = {
        "Authorization": f"Bearer {os.environ.get('LINE_NOTIFY_TOKEN')}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    payload = LineNotifyRequest(message=message)

    requests.post(
        url,
        headers=headers,
        data=payload.model_dump(exclude_none=True),
    )
    if image_paths:
        headers.pop("Content-Type")
        for i, image_path in enumerate(image_paths, 1):
            message = LineNotifyRequest(message=f"圖片{i}({image_path.name}):")
            with open(image_path, "rb") as f:
                print(
                    requests.post(
                        url,
                        data=message.model_dump(exclude_none=True),
                        headers=headers,
                        files={"imageFile": f},
                    ).json()
                )


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
    load_dotenv(override=True)
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
        with open(str(uuid4()) + ".json", "w", encoding="utf-8") as f:
            json.dump(self._history, f, ensure_ascii=False, indent=2)
        response = self.client.chat.completions.create(
            messages=self._history, **{**openai_config.model_dump(), **kwargs}
        )
        return response

    def ask(self, prompt: str | list[dict], **open_kwargs) -> tuple[str, CompletionUsage]:
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

    def visions(self, text, image_base64_list, **openai_kwargs) -> tuple[str, CompletionUsage]:
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

        response, token_usage = self.client.visions(text, image_base64_list, **openai_kwargs)
        return MuleProbModel.model_validate_json(response), token_usage
