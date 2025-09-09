from __future__ import annotations

import logging
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class UkrainianTranslator:
    def __init__(self, model_id: str = "tencent/Hunyuan-MT-7B"):
        self.model_id = model_id
        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        logger.info(f"Loading translation model: {self.model_id}")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                device_map="auto"
            )

            logger.info("Translation model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            raise RuntimeError(f"Translation model loading failed: {e}") from e

    def translate_to_ukrainian(self, text: str) -> str:
        """Translate text to Ukrainian using chat template."""
        if not text or not text.strip():
            return text

        self._ensure_model_loaded()

        try:
            messages = [
                {"role": "user", "content": f"Translate the following segment into Ukrainian, without additional explanation.\n\n{text}"},
            ]
            
            tokenized_chat = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt"
            )

            outputs = self._model.generate(
                tokenized_chat.to(self._model.device), 
                max_new_tokens=2048
            )
            
            output_text = self._tokenizer.decode(outputs[0])
            
            # Extract only the translated part (after the input prompt)
            # Find the assistant's response after the user message
            if "assistant" in output_text:
                parts = output_text.split("assistant")
                if len(parts) > 1:
                    translated_text = parts[-1].strip()
                    # Remove any special tokens or formatting
                    translated_text = translated_text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
                    return translated_text
            
            # Fallback: try to extract text after the original input
            if text in output_text:
                parts = output_text.split(text)
                if len(parts) > 1:
                    translated_text = parts[-1].strip()
                    # Clean up any remaining formatting
                    translated_text = translated_text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
                    return translated_text
            
            # If we can't parse properly, return the full output (might need manual cleanup)
            logger.warning("Could not parse translation output properly, returning full output")
            return output_text

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text