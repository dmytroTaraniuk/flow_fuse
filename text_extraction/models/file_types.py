from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


class FileType(Enum):
    IMAGE = "image"
    DIGITAL_DOCUMENT = "digital_document"
    SCANNED_DOCUMENT = "scanned_document"
    HYBRID_DOCUMENT = "hybrid_document"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass
class ExtractedImage:
    path: Path
    index: int
    source_file: Path
    page_number: Optional[int] = None
    
    
@dataclass
class ExtractionResult:
    source_file: Path
    file_type: FileType
    digital_text: str = ""
    ocr_text: str = ""
    combined_markdown: str = ""
    extracted_images: List[ExtractedImage] = field(default_factory=list)
    page_count: int = 0
    image_count: int = 0
    extraction_method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)