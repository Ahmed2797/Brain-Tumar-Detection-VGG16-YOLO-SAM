# Brain Tumar Detection

## 🛠️ Recommended Conda Environment

    conda create -n kidney python=3.12
    conda activate kidney

    # Install pip packages from requirements.txt
    
    pip install -r requirements.txt

## Download Dataset

    url = https://drive.google.com/file/d/1alECMHjosy7TBFWqFFMiY14bdFT-bEGC/view?usp=sharing

    🧠 1.YOLO-Ready Object Detection Dataset
    🔗 https://github.com/ultralytics/assets/releases/download/v0.0.0/brain-tumor.zip


    🧠 2. Binary Classification / Simple Detection MRI Images

    👉 Brain MRI Images for Brain Tumor Detection (Binary)
    🔗 https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
    
    tree -d

Separate Endpoints:

/predict_vgg: Uses VGG16 model for classification, returns JSON with prediction and confidence
/detect_yolo: Uses YOLO for detection, returns annotated image with bounding boxes and confidence labels
/segment_sam: Uses SAM for segmentation, returns annotated image with masks
Image Processing: All endpoints resize images to 512x512 for faster processing

Three Action Buttons:

"Predict VGG" - Shows classification result with confidence bar
"Detect YOLO" - Shows image with bounding boxes around detected tumors
"Segment SAM" - Shows image with precise segmentation masks
Image Display: Uploaded image is displayed first, then each prediction shows the annotated result image

Clean UI: Updated result cards for each model type

Key Features
Browse & Show Image: Upload form displays the selected image immediately
VGG16 Prediction: Returns text result (tumor/no tumor) with confidence percentage
YOLO Detection: Returns image with green bounding boxes and confidence scores
SAM Segmentation: Returns image with colored segmentation masks for easy understanding
The code is now clean, modular, and each prediction displays the processed image as requested. The frontend ensures users can easily understand the results through visual annotations.
