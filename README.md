# CDC OCR (Chendu City OCR)

A Python-based OCR (Optical Character Recognition) pipeline designed to detect, process, and extract text (and image regions) from documents or images with improved accuracy and flexibility. The project leverages [EasyOCR](https://github.com/JaidedAI/EasyOCR), OpenCV, and a customizable set of "handlers" to handle everything from rotation detection, text-like region clustering, and fewer false positives in text detection.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/cdc-ocr.git
cd cdc-ocr
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```


## Usage

Basic usage example:

```
cd preprocessing
python image_test.py
```


## Key Components

### Improved_Image_Preprocessor

The main class handling the OCR pipeline:

- Image loading and preprocessing
- Text block detection
- Image region detection
- Region clustering and merging

# TextDetector

Handles text detection with features:

- Multiple orientation support
- Duplicate text removal
- Confidence scoring

### Region Handlers

Collection of specialized handlers for:

- Block detection and grouping
- Region merging
- Layout analysis
- Visualization

## Output Format

The system produces structured output in JSON format containing:

- Detected regions with boundaries
- Text content with confidence scores
- Image regions with position information
- Hierarchical layout structure

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your chosen license]

## Acknowledgments

- EasyOCR for base OCR capabilities
- OpenCV for image processing
- [Other acknowledgments]
