import mimetypes
from pathlib import Path
import fitz
from ..core.interfaces import FileClassifier
from ..models.file_types import FileType
from docx import Document


class DefaultFileClassifier(FileClassifier):
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    DOCUMENT_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt', '.rtf', '.odt'}
    TEXT_EXTENSIONS = {'.txt', '.md', '.rst', '.log'}

    def classify(self, file_path: Path) -> FileType:
        if not file_path.exists():
            return FileType.UNKNOWN

        extension = file_path.suffix.lower()

        if extension in self.TEXT_EXTENSIONS:
            return FileType.TEXT

        if extension in self.IMAGE_EXTENSIONS:
            return FileType.IMAGE

        if extension == '.pdf':
            return self._classify_pdf(file_path)

        if extension in {'.docx', '.doc'}:
            return self._classify_office_doc(file_path)

        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            if mime_type.startswith('image/'):
                return FileType.IMAGE
            elif mime_type.startswith('text/'):
                return FileType.TEXT

        return FileType.UNKNOWN

    def _classify_pdf(self, file_path: Path) -> FileType:
        try:
            doc = fitz.open(str(file_path))
            has_text = False
            has_images = False

            for page in doc:
                text = page.get_text()
                if text.strip():
                    has_text = True

                image_list = page.get_images()
                if image_list:
                    has_images = True

                if has_text and has_images:
                    doc.close()
                    return FileType.HYBRID_DOCUMENT

            doc.close()

            if has_text and not has_images:
                return FileType.DIGITAL_DOCUMENT
            elif not has_text and has_images:
                return FileType.SCANNED_DOCUMENT
            elif has_text:
                return FileType.DIGITAL_DOCUMENT
            else:
                return FileType.UNKNOWN

        except Exception:
            return FileType.UNKNOWN

    def _classify_office_doc(self, file_path: Path) -> FileType:
        doc = Document(str(file_path))

        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                return FileType.HYBRID_DOCUMENT

        return FileType.DIGITAL_DOCUMENT
