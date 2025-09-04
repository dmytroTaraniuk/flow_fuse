# Flow Fuse

Text extraction system supporting multiple document formats with digital text extraction and OCR capabilities.

## Features

- **Multi-format support**: PDF, DOCX, images, plain text
- **Dual extraction**: Digital text + OCR for hybrid documents
- **Batch processing**: Process multiple files with parallel execution
- **Validation system**: Compare extraction results against validation markers
- **Image extraction**: Extract and process images from documents

## Quick Start

```bash
# Single file
python main.py document.pdf

# Multiple files with validation
python main.py *.pdf --validation-dir ./validations

# Batch processing with parallel jobs
python main.py folder/*.pdf --jobs 4 --output-file results.json
```

## Core Components

- **main.py**: CLI entry point with batch processing and validation
- **text_extraction/**: Core extraction framework
  - `extractors/`: Docling (digital) and HuggingFace OCR extractors
  - `processors/`: File processing pipeline
  - `classifiers/`: File type detection
  - `models/`: Data structures and types

## File Types

- `DIGITAL_DOCUMENT`: PDFs with selectable text
- `SCANNED_DOCUMENT`: Image-based PDFs requiring OCR
- `HYBRID_DOCUMENT`: Mixed digital + scanned content
- `IMAGE`: Direct image files
- `TEXT`: Plain text files

## CLI Options

- `--jobs N`: Parallel processing workers
- `--validation FILE`: Validation JSON file
- `--validation-dir DIR`: Auto-match validation files
- `--output-file FILE`: Save results as JSON
- `--ocr-quantization`: Memory optimization (none/8bit/4bit)
- `--verbose`: Detailed logging

## Dependencies

- docling: Digital document processing
- transformers: HuggingFace OCR models
- PyMuPDF: PDF handling
- tqdm: Progress bars