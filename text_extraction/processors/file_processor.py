import logging
from pathlib import Path
from typing import List, Optional, Dict, Type
import fitz
from ..core.interfaces import FileClassifier, TextExtractor, ImageExtractor
from ..models.file_types import FileType, ExtractionResult, ExtractedImage

logger = logging.getLogger(__name__)


class FileProcessor:
    
    def __init__(
        self,
        classifier: FileClassifier,
        text_extractors: List[TextExtractor],
        image_extractor: Optional[ImageExtractor] = None,
        output_dir: Optional[Path] = None
    ):
        self.classifier = classifier
        self.text_extractors = self._organize_extractors(text_extractors)
        self.image_extractor = image_extractor
        self.output_dir = output_dir or Path("extracted_images")
        logger.info(f"FileProcessor initialized with {len(text_extractors)} text extractors")
    
    def _organize_extractors(self, extractors: List[TextExtractor]) -> Dict[str, TextExtractor]:
        """Organize extractors by their purpose for easier access."""
        extractor_map = {}
        for extractor in extractors:
            class_name = extractor.__class__.__name__
            if "Docling" in class_name:
                extractor_map["digital"] = extractor
            elif "OCR" in class_name:
                extractor_map["ocr"] = extractor
        return extractor_map
        
    def process(self, file_path: Path) -> ExtractionResult:
        file_path = Path(file_path).resolve()
        logger.info(f"Starting processing of file: {file_path}")
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Classify file type
        file_type = self.classifier.classify(file_path)
        logger.info(f"File classified as: {file_type.value}")
        
        # Initialize result
        result = ExtractionResult(
            source_file=file_path,
            file_type=file_type,
            metadata={
                "file_size": file_path.stat().st_size,
                "file_name": file_path.name
            }
        )
        
        # Get page count for PDFs
        result.page_count = self._get_pdf_page_count(file_path)
        
        # Extract content based on file type
        self._extract_content(file_path, file_type, result)
        
        # Combine all text into markdown
        result.combined_markdown = self._combine_to_markdown(result)
        logger.info(f"Processing complete. Total text length: {len(result.combined_markdown)}")
        
        return result
    
    def _get_pdf_page_count(self, file_path: Path) -> int:
        """Get page count for PDF files."""
        if file_path.suffix.lower() != '.pdf':
            return 0
        
        try:
            with fitz.open(str(file_path)) as doc:
                page_count = len(doc)
            logger.info(f"PDF has {page_count} pages")
            return page_count
        except Exception as e:
            logger.warning(f"Could not get page count: {e}")
            return 0
    
    def _extract_content(self, file_path: Path, file_type: FileType, result: ExtractionResult) -> None:
        """Extract content based on file type."""
        if file_type == FileType.HYBRID_DOCUMENT:
            self._handle_hybrid_document(file_path, result)
        else:
            self._handle_single_type_document(file_path, file_type, result)
        
        # Extract and process images for all document types
        result.extracted_images = self._extract_images(file_path, file_type)
        result.image_count = len(result.extracted_images)
        logger.info(f"Extracted {result.image_count} images")
        
        # Process images for OCR if needed
        if result.extracted_images and file_type != FileType.HYBRID_DOCUMENT:
            self._process_images_for_ocr(result)
    
    def _handle_single_type_document(self, file_path: Path, file_type: FileType, result: ExtractionResult) -> None:
        """Handle documents with a single extraction type."""
        if file_type in [FileType.DIGITAL_DOCUMENT, FileType.TEXT]:
            logger.info("Attempting digital text extraction")
            result.digital_text = self._extract_text(file_path, file_type, "digital")
            if result.digital_text:
                result.extraction_method = "digital"
                logger.info(f"Extracted {len(result.digital_text)} characters of digital text")
        
        elif file_type in [FileType.IMAGE, FileType.SCANNED_DOCUMENT]:
            logger.info("Attempting OCR text extraction")
            result.ocr_text = self._extract_text(file_path, file_type, "ocr")
            if result.ocr_text:
                result.extraction_method = "ocr"
                logger.info(f"Extracted {len(result.ocr_text)} characters via OCR")
    
    def _handle_hybrid_document(self, file_path: Path, result: ExtractionResult) -> None:
        """Handle hybrid documents with both digital and scanned content."""
        logger.info("Hybrid document - attempting both extraction methods")
        
        # Extract digital text
        result.digital_text = self._extract_text(file_path, FileType.DIGITAL_DOCUMENT, "digital")
        
        # Extract and process images
        extracted_images = self._extract_images(file_path, FileType.HYBRID_DOCUMENT)
        if extracted_images:
            ocr_results = self._ocr_images(extracted_images)
            if ocr_results:
                result.ocr_text = "\n\n".join(ocr_results)
        
        result.extraction_method = "hybrid"
    
    def _extract_text(self, file_path: Path, file_type: FileType, extractor_type: str) -> str:
        """Extract text using the appropriate extractor."""
        extractor = self.text_extractors.get(extractor_type)
        if not extractor or not extractor.can_handle(file_type):
            logger.debug(f"No {extractor_type} extractor available or cannot handle {file_type}")
            return ""
        
        logger.info(f"Extracting text using {extractor.__class__.__name__} for {file_path}")
        try:
            result = extractor.extract(file_path)
            if result is None:
                logger.warning(f"{extractor_type.capitalize()} extractor returned None")
                return ""
            logger.info(f"{extractor_type.capitalize()} extraction successful, extracted {len(result)} characters")
            return result
        except Exception as e:
            logger.error(f"{extractor_type.capitalize()} extraction failed with {type(e).__name__}: {e}")
            # Re-raise the exception to see the full traceback in verbose mode
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return ""
    
    def _ocr_images(self, images: List[ExtractedImage]) -> List[str]:
        """OCR a list of images and return formatted results."""
        logger.info(f"Processing {len(images)} extracted images")
        results = []
        
        for img in images:
            text = self._extract_text(img.path, FileType.IMAGE, "ocr")
            if text:
                results.append(f"### Image {img.index + 1}\n{text}")
        
        return results
    
    def _process_images_for_ocr(self, result: ExtractionResult) -> None:
        """Process extracted images for OCR text extraction."""
        logger.info("Processing extracted images for text")
        processed_text = self._process_extracted_images(result.extracted_images)
        if processed_text:
            result.ocr_text = f"{result.ocr_text}\n\n{processed_text}" if result.ocr_text else processed_text
    
    def _extract_images(self, file_path: Path, file_type: FileType) -> List[ExtractedImage]:
        if not self.image_extractor:
            return []
        
        if self.image_extractor.can_extract_from(file_type):
            try:
                logger.debug(f"Extracting images from {file_path}")
                return self.image_extractor.extract_images(file_path, self.output_dir)
            except Exception as e:
                logger.warning(f"Image extraction failed: {e}")
                return []
        
        return []
    
    def _process_extracted_images(self, images: List[ExtractedImage]) -> str:
        """Process extracted images and OCR them."""
        results = []
        
        for image in images:
            logger.debug(f"Processing extracted image: {image.path}")
            text = self._extract_text(image.path, FileType.IMAGE, "ocr")
            if text:
                page_info = f" (Page {image.page_number})" if image.page_number else ""
                results.append(f"### Image {image.index + 1}{page_info}\n\n{text}")
        
        return "\n\n".join(results)
    
    def _combine_to_markdown(self, result: ExtractionResult) -> str:
        """Combine all extracted text into markdown format."""
        sections = []
        
        if result.digital_text:
            sections.append(f"## Digital Text Content\n\n{result.digital_text}")
        
        if result.ocr_text:
            sections.append(f"## OCR Extracted Content\n\n{result.ocr_text}")
        
        return "\n\n".join(sections) if sections else "No text content could be extracted from this file."