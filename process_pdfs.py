import os
import json
import torch
from ultralytics import YOLO
from pathlib import Path
import warnings
import fitz  # PyMuPDF
from PIL import Image
import io
import cv2
import numpy as np

try:
    import easyocr
    OCR_AVAILABLE = True
    print("✅ EasyOCR available")
except ImportError:
    try:
        import pytesseract
        OCR_AVAILABLE = True
        print("✅ Tesseract available")
    except ImportError:
        OCR_AVAILABLE = False
        print("⚠️ No OCR library available. Install easyocr or pytesseract for text extraction")

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

def load_model_safely(model_path):
    """Load YOLO model with proper error handling for PyTorch 2.6+"""
    try:
        # Method 1: Try with safe globals
        torch.serialization.add_safe_globals([
            'ultralytics.nn.tasks.YOLOv10DetectionModel',
            'ultralytics.nn.modules.head.Detect',
            'ultralytics.nn.modules.block.C2f',
            'ultralytics.nn.modules.conv.Conv',
            'ultralytics.nn.modules.block.SPPF',
            'collections.OrderedDict'
        ])
        model = YOLO(model_path)
        print("✅ Model loaded with safe globals")
        return model
    except Exception as e1:
        print(f"Method 1 failed: {e1}")
        
        try:
            # Method 2: Monkey patch torch.load temporarily
            original_load = torch.load
            def patched_load(*args, **kwargs):
                kwargs.pop('weights_only', None)  # Remove weights_only if present
                return original_load(*args, weights_only=False, **kwargs)
            
            torch.load = patched_load
            model = YOLO(model_path)
            torch.load = original_load  # Restore original
            print("✅ Model loaded with patched torch.load")
            return model
        except Exception as e2:
            print(f"All loading methods failed. Last error: {e2}")
            raise Exception(f"Could not load model: {e2}")

def extract_text_from_bbox(image, bbox, method='easyocr'):
    """Extract text from bounding box using OCR"""
    if not OCR_AVAILABLE:
        return "OCR not available"
    
    try:
        # Convert PIL to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Extract region of interest
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return "Empty region"
        
        # Use EasyOCR if available
        if method == 'easyocr' and 'easyocr' in globals():
            if not hasattr(extract_text_from_bbox, 'reader'):
                extract_text_from_bbox.reader = easyocr.Reader(['en'])
            
            results = extract_text_from_bbox.reader.readtext(roi)
            text = ' '.join([result[1] for result in results if result[2] > 0.5])  # confidence > 0.5
            return text.strip() if text.strip() else "No text detected"
        
        # Fallback to Tesseract
        elif 'pytesseract' in globals():
            text = pytesseract.image_to_string(roi, config='--psm 6').strip()
            return text if text else "No text detected"
        
        return "OCR failed"
    
    except Exception as e:
        return f"OCR error: {str(e)}"

def classify_heading_level(text, bbox, page_height, font_size_threshold=None):
    """
    Classify heading level based on text content and position
    This is a heuristic approach since your model doesn't detect H1, H2, etc.
    """
    text_lower = text.lower().strip()
    
    # Skip if text is too short or looks like body text
    if len(text) < 3 or len(text) > 200:
        return None
    
    # Calculate relative position in page (0-1)
    y_position = bbox[1] / page_height
    bbox_height = bbox[3] - bbox[1]
    
    # Heuristics for heading classification
    heading_indicators = [
        'chapter', 'section', 'introduction', 'conclusion', 'abstract',
        'methodology', 'results', 'discussion', 'references', 'appendix',
        'overview', 'summary', 'background', 'literature', 'method',
        'analysis', 'findings', 'future', 'related work'
    ]
    
    # Check if text contains heading-like words
    is_heading_like = any(indicator in text_lower for indicator in heading_indicators)
    
    # Check if text is short and title-case or uppercase
    is_title_case = text.istitle() or text.isupper()
    is_short = len(text.split()) <= 10
    
    # Determine heading level based on various factors
    if is_heading_like or is_title_case:
        if bbox_height > 50:  # Large text, likely H1
            return "H1"
        elif bbox_height > 35:  # Medium text, likely H2
            return "H2"
        elif bbox_height > 25:  # Smaller text, likely H3
            return "H3"
        else:
            return "H4"
    
    # Additional checks for numbered sections
    if text_lower.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
        return "H2"
    elif text_lower.startswith(('1.1', '1.2', '2.1', '2.2', '3.1', '3.2')):
        return "H3"
    
    return None

def pdf_to_images_pymupdf(pdf_path, output_dir, dpi=300):
    """Convert PDF to images using PyMuPDF (no Poppler required)"""
    images = []
    try:
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Create transformation matrix for desired DPI
            zoom = dpi / 72  # 72 is the default DPI
            mat = fitz.Matrix(zoom, zoom)
            
            # Render page to pixmap
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(img_data))
            
            # Save image
            img_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
            img.save(img_path, "PNG")
            images.append(img)
            print(f"Saved: {img_path}")
        
        doc.close()
        return images
        
    except Exception as e:
        print(f"Error converting PDF with PyMuPDF: {e}")
        return []

# Load YOLO model
try:
    model = load_model_safely("yolov10l_ft.pt")
except Exception as e:
    print(f"❌ Critical error: Could not load model: {e}")
    exit(1)

# Directories
INPUT_DIR = "sample_dataset/pdfs"
IMAGE_DIR = "sample_dataset/images"
OUTPUT_DIR = "sample_dataset/outputs"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Extract title from first page predictions
def extract_title(preds):
    for box in preds:
        if box['label'].lower() == 'title' and box.get('text', '').strip():
            return box['text']
    return "Untitled Document"

# Extract heading entries with proper classification
def extract_outline(preds, page_num, page_height):
    outline = []
    for box in preds:
        text = box.get('text', '').strip()
        if not text or text in ['No text detected', 'OCR not available']:
            continue
            
        # Try to classify as heading if model detected 'title' or if it looks like a heading
        heading_level = None
        if box['label'].lower() == 'title':
            # For title detections, try to classify the heading level
            heading_level = classify_heading_level(text, box['bbox'], page_height)
        
        if heading_level:
            outline.append({
                "level": heading_level,
                "text": text,
                "page": page_num,
                "confidence": box.get('confidence', 0),
                "bbox": box['bbox']
            })
    return outline

# Run YOLO on PDF and create JSON
def process_pdf(pdf_path):
    try:
        pdf_name = Path(pdf_path).stem
        print(f"Converting PDF to images: {pdf_name}")
        
        # Create subdirectory for this PDF's images
        pdf_image_dir = os.path.join(IMAGE_DIR, pdf_name)
        os.makedirs(pdf_image_dir, exist_ok=True)
        
        # Convert PDF to images
        images = pdf_to_images_pymupdf(pdf_path, pdf_image_dir)
        
        if not images:
            print("❌ Could not convert PDF to images")
            return
        
        all_boxes = []
        outline = []

        # Process each image
        for i, img in enumerate(images):
            img_path = os.path.join(pdf_image_dir, f"page_{i+1}.png")
            
            if not os.path.exists(img_path):
                print(f"❌ Image file not found: {img_path}")
                continue
                
            print(f"Running YOLO on page {i+1}")
            try:
                results = model(img_path, conf=0.4, verbose=False)
                page_preds = []

                # Load image for OCR
                cv_image = cv2.imread(img_path)
                page_height = cv_image.shape[0] if cv_image is not None else 1000

                # Check if any detections were found
                if hasattr(results[0], 'boxes') and results[0].boxes is not None and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        label = model.names[cls_id] if hasattr(model, 'names') else f"class_{cls_id}"
                        xyxy = box.xyxy[0].tolist()
                        confidence = float(box.conf[0])

                        # Extract actual text using OCR
                        if cv_image is not None and OCR_AVAILABLE:
                            extracted_text = extract_text_from_bbox(cv_image, xyxy)
                        else:
                            extracted_text = f"{label} Placeholder"

                        page_preds.append({
                            "label": label,
                            "text": extracted_text,
                            "bbox": xyxy,
                            "confidence": confidence
                        })
                    
                    print(f"Found {len(page_preds)} detections on page {i+1}")
                    
                    # Extract text from all detections and print sample
                    for pred in page_preds[:2]:  # Show first 2 detections
                        print(f"  {pred['label']}: '{pred['text'][:50]}...'")
                else:
                    print(f"No detections found on page {i+1}")

                all_boxes.append(page_preds)
                outline.extend(extract_outline(page_preds, i + 1, page_height))
                
            except Exception as yolo_error:
                print(f"❌ YOLO error on page {i+1}: {yolo_error}")
                all_boxes.append([])

        # Extract title from first page if available
        title = extract_title(all_boxes[0]) if all_boxes and all_boxes[0] else "Untitled Document"

        # Final JSON structure
        output_json = {
            "title": title,
            "total_pages": len(images),
            "outline": outline,
            #"all_detections": all_boxes
        }

        # Save result
        output_path = os.path.join(OUTPUT_DIR, f"{pdf_name}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_json, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved: {output_path}")
        print(f"📋 Extracted {len(outline)} headings")
        
    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {str(e)}")
        import traceback
        traceback.print_exc()

# Process all PDFs in input directory
def process_all_pdfs():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Input directory '{INPUT_DIR}' does not exist. Creating it...")
        os.makedirs(INPUT_DIR, exist_ok=True)
        print(f"Please place your PDF files in the '{INPUT_DIR}' directory and run again.")
        return
    
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print(f"❌ No PDF files found in '{INPUT_DIR}' directory.")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) to process:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file}")
    
    for file in pdf_files:
        print(f"\n🔍 Processing: {file}")
        process_pdf(os.path.join(INPUT_DIR, file))

if __name__ == "__main__":
    # Check if model file exists
    if not os.path.exists("yolov10l_ft.pt"):
        print("❌ Model file 'yolov10l_ft.pt' not found!")
        print("Please ensure the model file is in the current directory.")
    else:
        print("🔍 Model file found, starting processing...")
        
        # Check for required imports
        print(f"OCR Available: {OCR_AVAILABLE}")
        if not OCR_AVAILABLE:
            print("⚠️ Warning: No OCR library found. Install one of:")
            print("  pip install easyocr")
            print("  pip install pytesseract")
            
        process_all_pdfs()