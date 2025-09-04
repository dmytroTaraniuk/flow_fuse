from pathlib import Path
from typing import Optional


class Settings:
    DEFAULT_OUTPUT_DIR = Path("extracted_images")
    DEFAULT_OCR_MODEL = "Nanonets/nanonets-ocr"
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        ocr_model: Optional[str] = None,
        huggingface_token: Optional[str] = None
    ):
        self.output_dir = output_dir or self.DEFAULT_OUTPUT_DIR
        self.ocr_model = ocr_model or self.DEFAULT_OCR_MODEL
        self.huggingface_token = huggingface_token