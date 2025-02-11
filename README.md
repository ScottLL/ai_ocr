# Vision-OCR

A vision-based OCR pipeline designed to detect, process, and extract text from documents or images with improved accuracy and flexibility using vison llm.

 The project leverages [EasyOCR](https://github.com/JaidedAI/EasyOCR), OpenCV, and a customizable set of "handlers" to handle everything from rotation detection, text-like region clustering, and fewer false positives in text detection.

 Vision LLM applied in to the project which using better proformance ocr to help the affordable vision model (llama3.2-vision) with better proformance. 

---
![1739021232039](image/README/1739021232039.png)
## Installation

1. Clone the repository:

```bash
git clone https://github.com/ScottLL/cdc-ocr.git
cd cdc-ocr
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install Ollama:

For macOS or Linux:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

For Windows:
- Download and install from [Ollama.com](https://ollama.com)


4. Pull the LLaMA 3.2-Vision model:
```bash
ollama pull llama3.2-vision
```

5. start run the Ollama server
```bash 
ollama serve
```

* Note: In this project, we are using llama3.2-vision as the ocr's vision model, feel free to change it to other much more powerful vison/multimodal model for better preformance. 

## Usage

Basic usage example:

```bash
cd preprocessing
python image_test.py
```




## Key Components

### ImprovedImagePreprocessor
The core engine that orchestrates the entire document understanding pipeline:
- Intelligent document layout analysis
- Adaptive region detection for complex layouts
- Multi-scale processing for long documents
- Integration with EasyOCR and LLaMA 3.2-Vision
- Hierarchical structure preservation

### Text Processing Pipeline
Advanced text extraction and understanding:
- High-accuracy text detection using EasyOCR
- Context-aware text grouping
- Multi-language support
- Structural relationship preservation between text blocks
- Handling of various text orientations and layouts

### Vision Understanding System
Powered by LLaMA 3.2-Vision for comprehensive image analysis:
- Semantic understanding of image regions
- Visual content description generation
- Context-aware image-text relationship analysis
- Deep understanding of diagrams, charts, and complex visuals

### Region Analysis & Integration
Sophisticated region handling system:
- Smart segmentation of complex documents
- Meaningful region identification and classification
- Contextual relationship mapping between regions
- Adaptive handling of mixed content (text + images)
- Layout preservation for better understanding

## Output Format
The system generates a comprehensive JSON output including:
- Structured text content with position mapping
- Visual region descriptions powered by LLaMA 3.2-Vision
- Hierarchical document structure
- Confidence scores for extracted information
- Spatial relationships between elements
- Cross-references between text and visual elements
