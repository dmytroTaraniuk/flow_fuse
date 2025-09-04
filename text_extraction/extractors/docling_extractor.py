from pathlib import Path
from ..core.interfaces import TextExtractor
from ..models.file_types import FileType
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.pipeline.simple_pipeline import SimplePipeline


class DoclingTextExtractor(TextExtractor):
    
    SUPPORTED_TYPES = {
        FileType.DIGITAL_DOCUMENT,
        FileType.HYBRID_DOCUMENT,
        FileType.TEXT
    }
    
    def __init__(self):
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = False
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options.do_cell_matching = True
        self.pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4, device=AcceleratorDevice.AUTO
        )


        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options),
                InputFormat.DOCX: WordFormatOption(
                    pipeline_cls=SimplePipeline
                ),
            }
        )

    
    def can_handle(self, file_type: FileType) -> bool:
        return file_type in self.SUPPORTED_TYPES
    
    def extract(self, file_path: Path) -> str:
        try:
            result = self.converter.convert(str(file_path))
            return result.document.export_to_markdown()
        except Exception as e:
            raise Exception(f"Failed to extract text using Docling: {e}")