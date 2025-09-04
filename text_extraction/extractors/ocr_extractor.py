from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Literal, Dict, Any

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

from ..core.interfaces import TextExtractor
from ..models.file_types import FileType

logger = logging.getLogger(__name__)


class HuggingFaceOCRExtractor(TextExtractor):
    SUPPORTED_TYPES = {FileType.IMAGE, FileType.SCANNED_DOCUMENT}
    DEFAULT_MODEL = "nanonets/Nanonets-OCR-s"

    # Optimized prompt for structured document extraction
    DEFAULT_PROMPT = (
        "Extract the text from the above document as if you were reading it naturally. "
        "Return the tables in html format. Return the equations in LaTeX representation. "
        "If there is an image in the document and image caption is not present, add a small "
        "description of the image inside the <img></img> tag; otherwise, add the image caption "
        "inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL "
        "COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> "
        "or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes. "
        "If there is no text, tables, equations, or images in the document, return nothing."
    )

    def __init__(
            self,
            *,
            model_name: Optional[str] = None,
            prompt: Optional[str] = None,
            quantization: Optional[Literal["8bit", "4bit"]] = None,
            dtype: torch.dtype = torch.bfloat16,
            device_map: str = "auto",
            offload_folder: Optional[str] = None,
    ) -> None:

        self.model_name = model_name or self.DEFAULT_MODEL
        self.prompt = prompt or self.DEFAULT_PROMPT
        self.quantization = quantization
        self.dtype = dtype
        self.device_map = device_map
        self.offload_folder = offload_folder

        self._processor: Optional[AutoProcessor] = None
        self._model: Optional[AutoModelForVision2Seq] = None

        logger.info(f"Initialized OCR extractor with model: {self.model_name}")
        if quantization:
            logger.info(f"Quantization enabled: {quantization}")

    def can_handle(self, file_type: FileType) -> bool:
        return file_type in self.SUPPORTED_TYPES

    def extract(self, file_path: Path) -> str:
        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")

        logger.info(f"Extracting text from: {file_path}")

        self._ensure_model_loaded()

        image = self._load_image(file_path)
        messages = self._prepare_messages(image)
        return self._generate_text(messages, image)

    def _ensure_model_loaded(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        logger.info("Loading model and processor...")

        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            model_kwargs = self._build_model_config()
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                **model_kwargs
            )

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def _build_model_config(self) -> Dict[str, Any]:
        """Build model configuration with optimization settings."""
        config = {
            "device_map": self.device_map,
            "trust_remote_code": True,
        }

        if self.offload_folder:
            config["offload_folder"] = self.offload_folder

        if self.quantization in ("8bit", "4bit"):
            if BitsAndBytesConfig is None:
                raise RuntimeError(
                    "bitsandbytes required for quantization. Install with: pip install bitsandbytes"
                )
            config["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=(self.quantization == "8bit"),
                load_in_4bit=(self.quantization == "4bit"),
            )
            logger.info(f"Using {self.quantization} quantization")
        else:
            config["torch_dtype"] = self.dtype
            logger.info(f"Using dtype: {self.dtype}")

        return config

    def _load_image(self, image_path: Path) -> Image.Image:
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image
        except Exception as e:
            raise ValueError(f"Cannot load image {image_path}: {e}") from e

    def _prepare_messages(self, image: Image.Image) -> list:
        return [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": self.prompt},
            ],
        }]

    def _generate_text(self, messages: list, image: Image.Image) -> str:
        try:
            text_input = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self._processor(
                text=[text_input],
                images=[image],
                padding=True,
                return_tensors="pt"
            ).to(self._model.device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=2000,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self._processor.tokenizer.eos_token_id,
                )

            generated_ids = output_ids[0, inputs.input_ids.shape[1]:]
            result = self._processor.decode(generated_ids, skip_special_tokens=True)

            self._save_debug_output(result)

            logger.info(f"Generated {len(result)} characters of text")
            return result

        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise RuntimeError(f"Failed to generate text: {e}") from e

    def _save_debug_output(self, text: str) -> None:
        try:
            debug_file = Path("ocr_results_nanonets.md")
            debug_file.write_text(text, encoding="utf-8")
            logger.debug(f"Debug output saved to: {debug_file}")
        except Exception as e:
            logger.warning(f"Failed to save debug output: {e}")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model='{self.model_name}', "
            f"quantization={self.quantization}, "
            f"device_map='{self.device_map}')"
        )
