# 🧠 PDF Headings Extractor with YOLO and OCR

This project automates the extraction of **titles and headings** from PDF documents using a custom-trained **YOLOv10 model** and **OCR (EasyOCR / Tesseract)**. The result is a structured JSON file that contains the document title and outline, ready for analysis, indexing, or summarization.

---

## 🚀 Features

- ✅ Converts PDFs to high-resolution images (using **PyMuPDF** or **pdf2image**)
- ✅ Detects headings/titles using a custom **YOLOv10** model
- ✅ Extracts text from bounding boxes using **EasyOCR** or **Tesseract**
- ✅ Classifies heading levels (H1–H4) heuristically
- ✅ Outputs structured JSON with title and outline
- ✅ Runs fully offline (no internet needed after setup)

---

## 📁 Folder Structure

project/
├── Dockerfile
├── yolov10l_ft.pt # Custom trained YOLOv10 model (place this manually)
├── process_pdfs.py # Main processing script
├── sample_dataset/
│ ├── pdfs/ # Input PDFs
│ ├── images/ # Generated images per PDF
│ └── outputs/ # Final JSON outputs

yaml
Copy
Edit

---

## ⚙️ Requirements

Install required Python libraries:

```bash
pip install -r requirements.txt
```
Or manually install:

```bash
pip install ultralytics easyocr pytesseract opencv-python PyMuPDF pillow numpy
```

🐳 Docker Usage (Recommended)
🛠️ Build the Docker Image:
```bash
docker build --platform=linux/amd64 -t pdf-headings:v1 .
```

📦 Run the Docker Container:
```bash
docker run --rm \
  -v "$(pwd)/sample_dataset/pdfs:/app/sample_dataset/pdfs:ro" \
  -v "$(pwd)/sample_dataset/outputs:/app/sample_dataset/outputs" \
  -v "$(pwd)/sample_dataset/images:/app/sample_dataset/images" \
  --network none \
  pdf-headings:v1
```

🧠 How It Works
PDF Conversion: Each page of a PDF is converted to an image using PyMuPDF (preferred) or pdf2image.

YOLO Detection: A fine-tuned YOLOv10 model detects heading/title boxes in the image.

OCR Text Extraction: Extracts text from those boxes using EasyOCR or pytesseract.

Heading Classification: Classifies headings into H1, H2, H3, H4 based on heuristics like font size, position, and content.

JSON Output: Outputs a structured JSON file containing the title and outline.

📥 Input
Place your PDF files in:

```bash
sample_dataset/pdfs/
📤 Output
For each PDF, a corresponding JSON file is generated in:
```

```bash
sample_dataset/outputs/
Example output:
```

json
{
  "title": "Research on YOLOv10",
  "total_pages": 6,
  "outline": [
    {
      "level": "H1",
      "text": "Abstract",
      "page": 1,
      "confidence": 0.89,
      "bbox": [100, 120, 450, 170]
    },
    {
      "level": "H2",
      "text": "1. Introduction",
      "page": 2,
      "confidence": 0.84,
      "bbox": [110, 200, 480, 240]
    }
  ]
}
🧪 Testing PDF-to-Image Conversion
You can test standalone image conversion using:

```bash
python process_pdfs.py
Make sure "file03.pdf" exists in the same directory.
```
