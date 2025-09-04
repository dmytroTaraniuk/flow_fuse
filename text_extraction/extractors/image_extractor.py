from pathlib import Path
from typing import List
import fitz
from PIL import Image
from docx import Document
from docx.document import Document as DocumentType
import io
from ..core.interfaces import ImageExtractor
from ..models.file_types import FileType, ExtractedImage


class DefaultImageExtractor(ImageExtractor):
    
    def extract_images(self, file_path: Path, output_dir: Path) -> List[ExtractedImage]:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if file_path.suffix.lower() == '.pdf':
            return self._extract_from_pdf(file_path, output_dir)
        elif file_path.suffix.lower() in {'.docx', '.doc'}:
            return self._extract_from_docx(file_path, output_dir)
        else:
            return []
    
    def can_extract_from(self, file_type: FileType) -> bool:
        return file_type in {
            FileType.HYBRID_DOCUMENT,
            FileType.SCANNED_DOCUMENT,
            FileType.DIGITAL_DOCUMENT
        }
    
    def _extract_from_pdf(self, file_path: Path, output_dir: Path) -> List[ExtractedImage]:
        extracted_images = []
        
        try:
            doc = fitz.open(str(file_path))
            image_counter = 0
            
            for page_num, page in enumerate(doc):
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha > 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    
                    image_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(image_data))
                    
                    image_filename = f"{file_path.stem}_page{page_num+1}_img{image_counter+1}.png"
                    image_path = output_dir / image_filename
                    image.save(image_path)
                    
                    extracted_images.append(
                        ExtractedImage(
                            path=image_path,
                            index=image_counter,
                            source_file=file_path,
                            page_number=page_num + 1
                        )
                    )
                    
                    image_counter += 1
                    pix = None
            
            doc.close()
            
        except Exception as e:
            raise Exception(f"Failed to extract images from PDF: {e}")
        
        return extracted_images
    
    def _extract_from_docx(self, file_path: Path, output_dir: Path) -> List[ExtractedImage]:
        extracted_images = []
        
        try:
            doc = Document(str(file_path))
            image_counter = 0
            
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    image_data = rel.target_part.blob
                    
                    image = Image.open(io.BytesIO(image_data))
                    
                    ext = rel.target_ref.split('.')[-1]
                    if ext not in ['png', 'jpg', 'jpeg', 'gif', 'bmp']:
                        ext = 'png'
                    
                    image_filename = f"{file_path.stem}_img{image_counter+1}.{ext}"
                    image_path = output_dir / image_filename
                    image.save(image_path)
                    
                    extracted_images.append(
                        ExtractedImage(
                            path=image_path,
                            index=image_counter,
                            source_file=file_path
                        )
                    )
                    
                    image_counter += 1
                    
        except Exception as e:
            raise Exception(f"Failed to extract images from DOCX: {e}")
        
        return extracted_images