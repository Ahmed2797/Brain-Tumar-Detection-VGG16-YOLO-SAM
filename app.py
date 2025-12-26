from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import os
import tempfile
import cv2
import numpy as np
import io
from ultralytics import YOLO, SAM
from project.pipeline.prediction import ImagePredictor

app = FastAPI(
    title="Brain Tumor Detection API",
    description="API for brain tumor detection and segmentation",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Initialize models
MODEL_PATH = "final_model/model.keras"
vgg_predictor = ImagePredictor(MODEL_PATH)
yolo_model = YOLO("brain20.pt")
sam_model = SAM("sam_b.pt")

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

@app.get("/api")
async def api_status():
    return {"message": "Brain Tumor Detection API is running"}

@app.post("/predict_vgg")
async def predict_vgg(file: UploadFile = File(...)):
    """Predict tumor using VGG16 model."""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image file.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        label, confidence = vgg_predictor.predict(temp_path)

        return JSONResponse(content={
            "prediction": label,
            "confidence": confidence
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VGG prediction failed: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

@app.post("/detect_yolo")
async def detect_yolo(file: UploadFile = File(...)):
    """Detect tumors using YOLO model and return annotated image with bounding boxes."""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image file.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        # Resize for faster processing
        image = cv2.imread(temp_path)
        if image is not None:
            image = cv2.resize(image, (512, 512))
            cv2.imwrite(temp_path, image)

        # Run YOLO detection
        results = yolo_model.predict(temp_path)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = box.cls[0].item()
                if conf > 0.5:
                    detections.append({
                        "class": int(cls),
                        "confidence": conf,
                        "box": [x1, y1, x2, y2]
                    })

        # Create annotated image with boxes
        image = cv2.imread(temp_path)
        if image is not None:
            for detection in detections:
                x1, y1, x2, y2 = detection["box"]
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image, f"{detection['confidence']:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            success, img_encoded = cv2.imencode('.png', image)
            if success:
                return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/png")

        # Fallback
        with open(temp_path, 'rb') as f:
            return StreamingResponse(io.BytesIO(f.read()), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO detection failed: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

@app.post("/segment_sam")
async def segment_sam(file: UploadFile = File(...)):
    """Segment tumors using SAM model and return annotated image with masks."""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image file.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        # Resize for faster processing
        image = cv2.imread(temp_path)
        if image is not None:
            image = cv2.resize(image, (512, 512))
            cv2.imwrite(temp_path, image)

        # Run YOLO to get boxes
        results = yolo_model.predict(temp_path)
        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                if conf > 0.5:
                    boxes.append([x1, y1, x2, y2])

        # Run SAM segmentation
        if boxes:
            sam_results = sam_model.predict(temp_path, bboxes=boxes)
            sam_result = sam_results[0]
            if sam_result:
                annotated_img = sam_result.plot()
                success, img_encoded = cv2.imencode('.png', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
                if success:
                    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/png")

        # Fallback to original
        with open(temp_path, 'rb') as f:
            return StreamingResponse(io.BytesIO(f.read()), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SAM segmentation failed: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)