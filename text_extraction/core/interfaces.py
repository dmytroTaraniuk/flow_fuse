from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from ..models.file_types import FileType, ExtractionResult, ExtractedImage


class FileClassifier(ABC):
    @abstractmethod
    def classify(self, file_path: Path) -> FileType:
        pass


class TextExtractor(ABC):
    @abstractmethod
    def can_handle(self, file_type: FileType) -> bool:
        pass
    
    @abstractmethod
    def extract(self, file_path: Path) -> str:
        pass


class ImageExtractor(ABC):
    @abstractmethod
    def extract_images(self, file_path: Path, output_dir: Path) -> List[ExtractedImage]:
        pass
    
    @abstractmethod
    def can_extract_from(self, file_type: FileType) -> bool:
        pass