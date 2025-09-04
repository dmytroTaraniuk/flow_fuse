import argparse
import logging
import sys
from pathlib import Path
import json
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from text_extraction.classifiers.file_classifier import DefaultFileClassifier
from text_extraction.extractors.docling_extractor import DoclingTextExtractor
from text_extraction.extractors.ocr_extractor import HuggingFaceOCRExtractor
from text_extraction.extractors.image_extractor import DefaultImageExtractor
from text_extraction.processors.file_processor import FileProcessor
from text_extraction.config.settings import Settings
from text_extraction.utils.translation import UkrainianTranslator
from tqdm.auto import tqdm

# ---- existing setup_logging and compare() kept as-is ----

class TqdmHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            pass

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(level)
    h = TqdmHandler()           # <-- use tqdm-aware handler
    h.setLevel(level)
    h.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(h)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("docling").setLevel(logging.WARNING)
    return logging.getLogger(__name__)

def compare(results: Dict, validations: Dict):
    found, not_found = [], []
    for marker in validations.get('markers', []):
        if marker['value'] in results['extraction']['combined_markdown']:
            found.append(marker)
        else:
            not_found.append(marker)
    percentage = (len(found) / max(1, len(validations.get('markers', [])))) * 100
    return {
        'file': results['source_file'],
        'percent_found': percentage,
        'found': found,
        'not_found': not_found
    }

# ----------------------- helpers -----------------------

def expand_inputs(inputs: List[str]) -> List[Path]:
    """Expand globs and keep order, dedup by resolved path."""
    seen = set()
    out: List[Path] = []
    for pattern in inputs:
        matches = glob.glob(pattern)
        if not matches:
            # treat as literal
            matches = [pattern]
        for m in matches:
            p = Path(m).expanduser().resolve()
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out

def load_json(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_validation_mapping(
    inputs: List[Path],
    single_or_list: List[Path],
    map_file: Optional[Path],
    val_dir: Optional[Path],
    map_by_order: bool,
) -> Dict[Path, Optional[Path]]:
    """
    Decide which validation JSON goes with which input.
    Priority (last one wins): single/list, map_file, val_dir
    """
    mapping: Dict[Path, Optional[Path]] = {p: None for p in inputs}

    # 1) single_or_list
    if single_or_list:
        if len(single_or_list) == 1 and not map_by_order:
            # one-for-all
            for p in inputs:
                mapping[p] = single_or_list[0]
        elif map_by_order:
            if len(single_or_list) != len(inputs):
                raise ValueError("--map-by-order requires same count of --validation as inputs")
            for p, v in zip(inputs, single_or_list):
                mapping[p] = v
        else:
            # heuristics: try to match by basename stem
            by_stem = {v.stem: v for v in single_or_list}
            for p in inputs:
                mapping[p] = by_stem.get(p.stem, mapping[p])

    # 2) map_file overrides
    if map_file:
        m = load_json(map_file)
        # keys may be basenames or full paths
        for p in inputs:
            key_candidates = [str(p), str(p.resolve()), p.name, p.stem]
            for k in key_candidates:
                if k in m:
                    mapping[p] = Path(m[k]).expanduser().resolve()
                    break

    # 3) val_dir overrides (auto-discovery)
    if val_dir:
        for p in inputs:
            candidates = [
                val_dir / f"{p.stem}.json",
                val_dir / f"{p.stem}.validation.json"
            ]
            for c in candidates:
                if c.exists():
                    mapping[p] = c.resolve()
                    break

    return mapping

def process_one_file(
    input_path: Path,
    settings: Settings,
    classifier: DefaultFileClassifier,
    text_extractors,
    image_extractor: DefaultImageExtractor,
    validation_path: Optional[Path],
    logger: logging.Logger,
    generate_description: bool = False
) -> Dict:
    """
    Runs the extraction pipeline for one file and (optionally) attaches validation.
    """
    processor = FileProcessor(
        classifier=classifier,
        text_extractors=text_extractors,
        image_extractor=image_extractor,
        output_dir=settings.output_dir
    )

    result = processor.process(input_path)

    output = {
        "status": "success",
        "source_file": str(result.source_file),
        "file_info": {
            "type": result.file_type.value,
            "size": result.metadata.get("file_size", 0),
            "name": result.metadata.get("file_name", ""),
            "page_count": result.page_count,
            "image_count": result.image_count
        },
        "extraction": {
            "method": result.extraction_method,
            "digital_text": {
                "extracted": bool(result.digital_text),
                "length": len(result.digital_text) if result.digital_text else 0,
                "content": result.digital_text
            },
            "ocr_text": {
                "extracted": bool(result.ocr_text),
                "length": len(result.ocr_text) if result.ocr_text else 0,
                "content": result.ocr_text
            },
            "combined_markdown": result.combined_markdown
        },
        "extracted_images": [
            {
                "index": img.index,
                "path": str(img.path),
                "page_number": img.page_number,
                "filename": img.path.name
            }
            for img in result.extracted_images
        ],
        "statistics": {
            "total_text_length": len(result.combined_markdown),
            "digital_text_length": len(result.digital_text) if result.digital_text else 0,
            "ocr_text_length": len(result.ocr_text) if result.ocr_text else 0,
            "has_content": bool(result.combined_markdown and result.combined_markdown != "No text content could be extracted from this file.")
        }
    }
    
    # Add image description if requested and this is an image file
    if generate_description and result.file_type.value in ['image', 'scanned_document']:
        try:
            # Find the OCR extractor that supports description
            ocr_extractor = None
            for extractor in text_extractors:
                if hasattr(extractor, 'generate_description') and extractor.generate_description:
                    ocr_extractor = extractor
                    break
            
            if ocr_extractor:
                original_desc, ukrainian_desc = ocr_extractor.generate_image_description(input_path)
                output["image_description"] = {
                    "original": original_desc,
                    "ukrainian": ukrainian_desc
                }
                logger.info(f"Generated image description for {input_path}")
        except Exception as e:
            logger.warning(f"Failed to generate image description for {input_path}: {e}")
            output["image_description"] = {
                "original": "",
                "ukrainian": "",
                "error": str(e)
            }

    if validation_path and validation_path.exists():
        validations = load_json(validation_path)
        comp = compare(output, validations)
        output["validation"] = comp

    return output

# ----------------------- main -----------------------

def main():
    parser = argparse.ArgumentParser(description="Extract text from various file formats (batch-ready)")
    # inputs accept multiple and globs
    parser.add_argument("inputs", nargs="+", help="Path(s) or glob(s) to input files")
    parser.add_argument("--output-dir", type=str, default="extracted_images", help="Directory for extracted images")
    parser.add_argument("--output-file", type=str, help="Save output to file (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    # validation options
    parser.add_argument("--validation", action="append", default=[], help="Validation JSON file (repeatable). One file applies to all, or use --map-by-order for 1:1 pairing.")
    parser.add_argument("--validation-map", type=str, help="Path to a JSON mapping: {'basename_or_path': 'path/to/validation.json', ...}")
    parser.add_argument("--validation-dir", type=str, help="Directory to auto-match <stem>.json or <stem>.validation.json")
    parser.add_argument("--map-by-order", action="store_true", help="Pair inputs and --validation files by order (counts must match).")

    # quality of life
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers for batch processing")

    # progress bar
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")

    parser.add_argument(
        "--ocr-quantization",
        choices=["none", "8bit", "4bit"],
        default="none",
        help="Quantization mode for the nanonets backend. "
             "Requires bitsandbytes when set to 8bit/4bit."
    )

    parser.add_argument(
        "--ocr-model",
        type=str,
        default="nanonets/Nanonets-OCR-s",
        help="Hugging Face model repo to use for nanonets backend."
    )


    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device mapping strategy for model placement (default: auto)"
    )

    parser.add_argument(
        "--offload-folder",
        type=str,
        default="./offload",
        help="Folder path for disk offloading to save memory (default: ./offload)"
    )

    parser.add_argument(
        "--description",
        action="store_true",
        help="Generate image descriptions using OCR model and translate to Ukrainian. Results will include both original and translated descriptions."
    )

    args = parser.parse_args()

    show_progress = (not args.no_progress) and (tqdm is not None)

    # Setup logging
    logger = setup_logging(args.verbose)
    logger.info("=== Text Extraction System Started (batch) ===")

    logger.info("OCR backend: nanonets")
    logger.info(f"OCR quantization: {args.ocr_quantization}")
    logger.info(f"Device mapping: {args.device_map}")
    logger.info(f"Offload folder: {args.offload_folder}")

    # Settings
    settings = Settings(
        output_dir=Path(args.output_dir),
        ocr_model=args.ocr_model
    )
    logger.info(f"Output directory: {settings.output_dir}")
    logger.info(f"OCR model: {settings.ocr_model}")

    # Expand inputs
    input_paths = expand_inputs(args.inputs)
    if not input_paths:
        print(json.dumps({"status": "error", "error": "No input files found"}, indent=2), file=sys.stderr)
        return 1
    logger.info(f"Resolved {len(input_paths)} input file(s).")

    # Prepare validation mapping
    single_or_list = [Path(v).expanduser().resolve() for v in args.validation]
    map_file = Path(args.validation_map).expanduser().resolve() if args.validation_map else None
    val_dir = Path(args.validation_dir).expanduser().resolve() if args.validation_dir else None

    try:
        validation_mapping = build_validation_mapping(
            inputs=input_paths,
            single_or_list=single_or_list,
            map_file=map_file,
            val_dir=val_dir,
            map_by_order=args.map_by_order
        )
    except Exception as e:
        print(json.dumps({"status": "error", "error": f"Validation mapping error: {e}"}, indent=2), file=sys.stderr)
        return 1

    # Initialize reusables
    classifier = DefaultFileClassifier()

    quant = None if args.ocr_quantization == "none" else args.ocr_quantization
    logger.info(f"Using OCR model: {args.ocr_model}")

    ocr_extractor = HuggingFaceOCRExtractor(
        model_name=args.ocr_model,
        quantization=quant,
        device_map=args.device_map,
        offload_folder=args.offload_folder,
        generate_description=args.description,
    )

    text_extractors = [DoclingTextExtractor(), ocr_extractor]
    image_extractor = DefaultImageExtractor()

    # Process
    results = []
    errors = []

    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {
                ex.submit(
                    process_one_file,
                    p,
                    settings,
                    classifier,
                    text_extractors,
                    image_extractor,
                    validation_mapping.get(p),
                    logger,
                    args.description
                ): p for p in input_paths
            }
            # Progress bar for parallel branch: advance when each task finishes
            if show_progress:
                pbar = tqdm(total=len(input_paths), desc="Processing", unit="file", dynamic_ncols=True, leave=True, position=0)
            else:
                pbar = None

            try:
                for fut in as_completed(futs):
                    p = futs[fut]
                    try:
                        results.append(fut.result())
                    except Exception as e:
                        errors.append({"file": str(p), "error": str(e)})
                    finally:
                        if pbar:
                            pbar.update(1)
                            pbar.refresh()
            finally:
                if pbar:
                    pbar.close()
    else:
        # Single-thread: iterate with tqdm over inputs
        iterable = input_paths
        if show_progress:
            iterable = tqdm(iterable, desc="Processing", unit="file", dynamic_ncols=True, leave=True, position=0)
        for p in iterable:
            try:
                results.append(
                    process_one_file(
                        p, settings, classifier, text_extractors, image_extractor, validation_mapping.get(p), logger, args.description
                    )
                )
            except Exception as e:
                errors.append({"file": str(p), "error": str(e)})

    # Make a top-level summary
    summary = {
        "total": len(input_paths),
        "succeeded": sum(1 for r in results if r.get("status") == "success"),
        "failed": len(errors)
    }

    # (Optional) aggregate validation stats if present
    validated = [r for r in results if "validation" in r]
    if validated:
        avg_found = sum(r["validation"]["percent_found"] for r in validated) / len(validated)
        summary["validation"] = {
            "validated_files": len(validated),
            "avg_percent_found": avg_found
        }

    payload = {
        "status": "partial" if errors and results else ("error" if errors and not results else "success"),
        "results": results,
        "errors": errors,
        "summary": summary
    }

    output_text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"Results saved to {args.output_file}")
    else:
        print(output_text)

    return 0 if not errors else 2

if __name__ == "__main__":
    exit(main())
