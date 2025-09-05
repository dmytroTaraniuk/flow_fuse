from __future__ import annotations

import logging
import re
from typing import Optional, List
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class UkrainianTranslator:
    def __init__(self, model_id: str = "facebook/nllb-200-distilled-1.3B", max_batch_length: int = 100):
        self.model_id = model_id
        self.max_batch_length = max_batch_length  # Maximum characters per batch
        self._model: Optional[AutoModelForSeq2SeqLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        logger.info(f"Loading translation model: {self.model_id}")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, src_lang="eng_Latn")

            bnb = BitsAndBytesConfig(load_in_8bit=True)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id,
                quantization_config=bnb,
                device_map="auto"
            )

            logger.info("Translation model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            raise RuntimeError(f"Translation model loading failed: {e}") from e

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns."""
        # More comprehensive sentence splitting patterns
        # Split on sentence endings followed by whitespace and capital letter, or paragraph breaks
        patterns = [
            r'(?<=[.!?])\s+(?=[A-Z])',  # Standard sentence endings
            r'(?<=\.)\s+(?=[A-Z][a-z])',  # Period followed by capitalized word
            r'\n\n+',  # Paragraph breaks
            r'(?<=\.)\s+(?=The |In |They |It |This |That |He |She |We |You |There |Here )',  # Common sentence starters
        ]
        
        # Apply multiple splitting patterns
        sentences = [text.strip()]
        for pattern in patterns:
            new_sentences = []
            for sentence in sentences:
                new_sentences.extend(re.split(pattern, sentence))
            sentences = new_sentences

        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 5:  # Ignore very short fragments
                cleaned_sentences.append(sentence)

        # If we couldn't split well, try a simpler approach
        if len(cleaned_sentences) <= 1 and len(text) > self.max_batch_length:
            # Split by periods more aggressively for long texts
            parts = text.split('. ')
            cleaned_sentences = []
            for i, part in enumerate(parts):
                part = part.strip()
                if part:
                    # Add period back except for last part (unless it already ends with punctuation)
                    if i < len(parts) - 1 and not part.endswith(('.', '!', '?')):
                        part += '.'
                    cleaned_sentences.append(part)

        return cleaned_sentences if cleaned_sentences else [text]

    def _create_batches(self, sentences: List[str]) -> List[str]:
        """Create intelligent batches from sentences, respecting max_batch_length."""
        if not sentences:
            return []

        batches = []
        current_batch = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence would exceed max length and we have content
            if current_length + sentence_length + 1 > self.max_batch_length and current_batch:
                # Finish current batch
                batches.append(' '.join(current_batch))
                current_batch = [sentence]
                current_length = sentence_length
            else:
                # Add sentence to current batch
                current_batch.append(sentence)
                current_length += sentence_length + (1 if current_batch else 0)  # +1 for space

            # Handle extremely long sentences that exceed max_batch_length
            if sentence_length > self.max_batch_length:
                # Split long sentence by chunks, preferring word boundaries
                if current_batch and len(current_batch) > 1:
                    # Remove the long sentence from current batch and finish it
                    current_batch.pop()
                    batches.append(' '.join(current_batch))

                # Split the long sentence into chunks
                words = sentence.split()
                chunk_words = []
                chunk_length = 0

                for word in words:
                    word_length = len(word)
                    if chunk_length + word_length + 1 > self.max_batch_length and chunk_words:
                        batches.append(' '.join(chunk_words))
                        chunk_words = [word]
                        chunk_length = word_length
                    else:
                        chunk_words.append(word)
                        chunk_length += word_length + (1 if len(chunk_words) > 1 else 0)

                # Add remaining words
                if chunk_words:
                    current_batch = chunk_words
                    current_length = chunk_length
                else:
                    current_batch = []
                    current_length = 0

        # Add final batch if it has content
        if current_batch:
            batches.append(' '.join(current_batch))

        return batches

    def _translate_batch(self, batch_text: str) -> str:
        """Translate a single batch of text."""
        try:
            # Estimate tokens needed (rough approximation: chars / 4)
            estimated_tokens = len(batch_text) // 4
            max_tokens = max(estimated_tokens * 2, 500)  # At least 500, usually 2x input length
            
            inputs = self._tokenizer(batch_text, return_tensors="pt", truncation=True, max_length=512).to(self._model.device)

            generated = self._model.generate(
                **inputs,
                forced_bos_token_id=self._tokenizer.convert_tokens_to_ids("ukr_Cyrl"),
                max_new_tokens=max_tokens,
                min_length=10,  # Ensure we get some output
                num_beams=3,    # Reduce beams for faster processing
                no_repeat_ngram_size=2,  # Reduce to allow some repetition
                early_stopping=False,    # Don't stop early
                do_sample=False,         # Deterministic output
                length_penalty=1.0,      # No length penalty
                pad_token_id=self._tokenizer.eos_token_id
            )

            result = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            
            # Clean up result - remove source language prefix if present
            if result.startswith(batch_text):
                result = result[len(batch_text):].strip()

            return result.strip()

        except Exception as e:
            logger.warning(f"Failed to translate batch (length {len(batch_text)}): {e}")
            return batch_text  # Return original text if translation fails

    def translate_to_ukrainian(self, text: str) -> str:
        """Translate text to Ukrainian using intelligent batching."""
        if not text or not text.strip():
            return text

        self._ensure_model_loaded()

        try:
            # For short texts, translate directly
            if len(text) <= self.max_batch_length:
                logger.info(f"Translating short text directly ({len(text)} chars, limit: {self.max_batch_length})")
                return self._translate_batch(text)

            # For longer texts, use intelligent batching
            logger.info(f"Translating long text ({len(text)} chars) using intelligent batching")

            sentences = self._split_into_sentences(text)
            batches = self._create_batches(sentences)

            logger.info(f"Split into {len(sentences)} sentences and {len(batches)} batches")

            translated_batches = []
            for i, batch in enumerate(batches):
                logger.debug(f"Translating batch {i+1}/{len(batches)} ({len(batch)} chars)")
                translated_batch = self._translate_batch(batch)
                logger.debug(f"Batch to translate: '{batch}'")
                logger.debug(f"Translated batch: '{translated_batch}'")
                translated_batches.append(translated_batch)

            result = ' '.join(translated_batches)
            logger.info(f"Translated text from {len(text)} to {len(result)} characters using {len(batches)} batches")

            return result

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text
