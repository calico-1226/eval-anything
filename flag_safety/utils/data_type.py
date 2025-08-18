import base64
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import requests
from openai.types.chat import ChatCompletionRole
from PIL import Image

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def split(input_string: str, separator: str = "<image>") -> List[str]:
    """
    Split a string by separator tags while preserving the separators as elements.

    Args:
        input_string (str): The input string to split
        separator (str): The separator to look for (default: "<image>")

    Returns:
        List[str]: A list with separators and content as separate elements

    Example:
        Input: "<image>content1<image><image>content2<image>"
        Output: ["<image>", "content1", "<image>", "<image>", "content2", "<image>"]
    """
    if not separator:
        return [input_string] if input_string else []

    result = []
    current_text = ""
    separator_length = len(separator)

    i = 0
    while i < len(input_string):
        if (
            i + separator_length <= len(input_string)
            and input_string[i : i + separator_length] == separator
        ):
            if current_text:
                result.append(current_text)
                current_text = ""
            result.append(separator)
            i += separator_length
        else:
            current_text += input_string[i]
            i += 1

    if current_text:
        result.append(current_text)

    return result


@dataclass
class EvaluationItem:
    conversation: List[Dict[str, Any]]
    is_valid: bool

    @classmethod
    def init_from_text(
        cls,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        is_valid: bool = True,
    ):
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
        ]
        if system_prompt:
            conversation.insert(
                0,
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
            )
        return cls(conversation=conversation, is_valid=is_valid)

    def add_text_content(self, text: str, role: ChatCompletionRole = "assistant"):
        try:
            self.conversation.append(
                {"role": role, "content": [{"type": "text", "text": text}]}
            )
        except:
            self.is_valid = False

    @classmethod
    def init_from_image_text(
        cls,
        user_prompt: str,
        images: List[Union[str, Image.Image]],
        system_prompt: Optional[str] = None,
        image_separator: str = "<image>",
        is_valid: bool = True,
    ):
        try:
            if isinstance(images, str) or isinstance(images, Image.Image):
                images = [images]
            image_placeholder_count = user_prompt.count(image_separator)
            images_count = len(images)

            text_segments = split(user_prompt, separator=image_separator)

            if image_placeholder_count < images_count:
                if not (image_placeholder_count == 0 and images_count == 1):
                    logger.warning(
                        f"Mismatch between image placeholders ({image_placeholder_count}) and images ({images_count}), extra images will be added to the end of user prompt. The user prompt is: {user_prompt}"
                    )
                for _ in range(images_count - image_placeholder_count):
                    text_segments.append(image_separator)
            elif image_placeholder_count > images_count and images_count > 0:
                # example: ["<image>", <image>", "content1", "<image>", "<image>", "content2", "<image>"]
                # if images_count = 1, then the new_segments will be ["<image>"]
                # if images_count = 2, then the new_segments will be ["<image>", "<image>", "content1"]
                # if images_count = 3, then the new_segments will be ["<image>", "<image>", "content1", "<image>"]
                # if images_count = 4, then the new_segments will be ["<image>", "<image>", "content1", "<image>", "<image>", "content2"]
                logger.warning(
                    f"Mismatch between image placeholders ({image_placeholder_count}) and images ({images_count}), extra content will be removed from the end of user prompt. The user prompt is: {user_prompt}"
                )
                new_segments = []
                image_count_seen = 0
                for i, segment in enumerate(text_segments):
                    new_segments.append(segment)
                    if segment == image_separator:
                        image_count_seen += 1
                        if image_count_seen == images_count:
                            if (
                                i + 1 < len(text_segments)
                                and text_segments[i + 1] != image_separator
                            ):
                                new_segments.append(text_segments[i + 1])
                            break
                text_segments = new_segments

            content = []
            image_idx = 0
            for text in text_segments:
                if text == image_separator:
                    if image_idx < len(images):
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": cls.encode_image(images[image_idx])
                                },
                            }
                        )
                        image_idx += 1
                else:
                    content.append({"type": "text", "text": text})

            conversation = [{"role": "user", "content": content}]
            if system_prompt:
                conversation.insert(0, {"role": "system", "content": system_prompt})

            return cls(conversation=conversation, is_valid=is_valid)
        except:
            return cls(conversation=[], is_valid=False)

    def add_image_content(
        self,
        images: List[Union[str, Image.Image]],
        role: ChatCompletionRole = "assistant",
    ):
        try:
            if isinstance(images, str) or isinstance(images, Image.Image):
                images = [images]
            content = []
            for img in images:
                content.append(
                    {"type": "image_url", "image_url": {"url": self.encode_image(img)}}
                )
            self.conversation.append({"role": role, "content": content})
        except:
            self.is_valid = False

    # TODO: Process the case that the number of image_separator is not equal to the number of images
    def add_text_image_content(
        self,
        text: str,
        images: List[Union[str, Image.Image]],
        role: ChatCompletionRole = "assistant",
        image_separator: str = "<image>",
    ):
        try:
            if isinstance(images, str) or isinstance(images, Image.Image):
                images = [images]
            text_segments = split(text, separator=image_separator)
            content = []
            image_idx = 0
            for text in text_segments:
                if text == image_separator:
                    if image_idx < len(images):
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": self.encode_image(images[image_idx])
                                },
                            }
                        )
                        image_idx += 1
                else:
                    content.append({"type": "text", "text": text})
            self.conversation.append({"role": role, "content": content})
        except:
            self.is_valid = False

    @classmethod
    def init_from_video_text(
        cls,
        user_prompt: str,
        videos: List[Union[str, List[Image.Image]]],
        system_prompt: Optional[str] = None,
        video_separator: str = "<video>",
        is_valid: bool = True,
    ):
        raise NotImplementedError("Video is not supported yet")

    @classmethod
    def init_from_audio_text(
        cls,
        user_prompt: str,
        audios: List[Union[str, np.ndarray, List[np.ndarray]]],
        system_prompt: Optional[str] = None,
        audio_separator: str = "<audio>",
        is_valid: bool = True,
    ):
        raise NotImplementedError("Audio is not supported yet")

    @classmethod
    def init_from_image_audio_text(
        cls,
        user_prompt: str,
        images: List[Union[str, Image.Image]],
        audios: List[Union[str, np.ndarray, List[np.ndarray]]],
        system_prompt: Optional[str] = None,
        image_separator: str = "<image>",
        audio_separator: str = "<audio>",
        is_valid: bool = True,
    ):
        raise NotImplementedError("Image and audio is not supported yet")

    @classmethod
    def init_from_image_video_text(
        cls,
        user_prompt: str,
        images: List[Union[str, Image.Image]],
        videos: List[Union[str, List[Image.Image]]],
        system_prompt: Optional[str] = None,
        image_separator: str = "<image>",
        video_separator: str = "<video>",
        is_valid: bool = True,
    ):
        raise NotImplementedError("Image and video is not supported yet")

    @classmethod
    def init_from_video_audio_text(
        cls,
        user_prompt: str,
        videos: List[Union[str, List[Image.Image]]],
        audios: List[Union[str, np.ndarray, List[np.ndarray]]],
        system_prompt: Optional[str] = None,
        video_separator: str = "<video>",
        audio_separator: str = "<audio>",
        is_valid: bool = True,
    ):
        raise NotImplementedError("Video and audio is not supported yet")

    @classmethod
    def init_from_image_video_audio_text(
        cls,
        user_prompt: str,
        images: List[Union[str, Image.Image]],
        videos: List[Union[str, List[Image.Image]]],
        audios: List[Union[str, np.ndarray, List[np.ndarray]]],
        system_prompt: Optional[str] = None,
        image_separator: str = "<image>",
        video_separator: str = "<video>",
        audio_separator: str = "<audio>",
        is_valid: bool = True,
    ):
        raise NotImplementedError("Image, video and audio is not supported yet")

    # TODO: Support more multimodal data types
    # FIXME: When printing, decode base64 multimodal data for display (currently buggy)
    def __str__(self):
        conversation = []
        for message in self.conversation:
            if message["type"] == "image_url":
                message["image_url"]["url"] = self.decode_image(
                    message["image_url"]["url"]
                )
            conversation.append(message)
        return str({"conversation": conversation, "is_valid": self.is_valid})

    # FIXME: Consider the situation that the image is RGBA or other format that needs more processing
    @staticmethod
    def encode_image(image: Union[str, Image.Image]) -> str:
        """
        Encode an image as a base64 data URL.

        Args:
            image: Path to an image or a PIL Image object

        Returns:
            Base64 encoded image as a PIL Image object
        """
        if isinstance(image, str):
            if image.startswith(("http://", "https://")):
                response = requests.get(image)
                image_input = Image.open(BytesIO(response.content))
            else:
                image_input = Image.open(image)
        else:
            image_input = image

        if image_input.mode != "RGB":
            image_input = image_input.convert("RGB")

        buffer = BytesIO()
        image_input.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()
        base64_data = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_data}"

    @staticmethod
    def decode_image(encoded_image: str) -> Image.Image:
        """
        Decode a base64 data URL to a PIL Image object.

        Args:
            encoded_image: The base64 data URL to decode

        Returns:
            A PIL Image object
        """
        if encoded_image.startswith("data:image"):
            encoded_image = encoded_image.split(",")[1]

        image_bytes = base64.b64decode(encoded_image)
        image = Image.open(BytesIO(image_bytes))
        return image
