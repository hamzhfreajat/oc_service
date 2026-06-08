from fastapi import FastAPI, UploadFile, File, HTTPException
import easyocr
import numpy as np
from PIL import Image
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OCR Watermark Service")

# Load model globally on startup (downloads weights if missing)
logger.info("Loading EasyOCR Model...")
# Use quantize=True for significantly faster CPU inference
READER = easyocr.Reader(['en', 'ar'], gpu=False, quantize=True)
logger.info("EasyOCR Model loaded successfully.")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/detect")
async def detect_watermark(file: UploadFile = File(...)):
    """
    Expects an image file upload.
    Returns: {"has_watermark": bool, "detections": [...]}
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be an image.")
        
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert('RGB')
        
        # Resize image to speed up CPU inference dramatically
        # 800x800 is enough to read watermarks effectively
        img.thumbnail((800, 800))
        
        img_np = np.array(img)
        
        # Read text from image
        results = READER.readtext(img_np)
        
        detections = []
        has_watermark = False
        
        for (bbox, text, prob) in results:
            clean_text = text.strip()
            # If we find highly confident, decent-sized text, we flag it as a watermark
            if prob > 0.6 and len(clean_text) >= 4:
                has_watermark = True
                detections.append({"text": clean_text, "prob": float(prob)})
                
        return {
            "has_watermark": has_watermark,
            "detections": detections
        }
        
    except Exception as e:
        logger.error(f"Failed to process image OCR: {e}")
        raise HTTPException(status_code=500, detail="Error processing image OCR")
