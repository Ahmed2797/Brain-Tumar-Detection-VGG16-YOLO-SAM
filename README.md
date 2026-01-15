# 🧠 Brain Tumor Detection WebApp

An AI-powered **Brain Tumor Detection Application** that provides classification, detection, and segmentation of MRI images using **VGG16, YOLO, and SAM models**.

The project is modular, clean, and designed to give a clear visual understanding of predictions.

![CI](https://github.com/Ahmed2797/Brain-Tumar-Detection-VGG16-YOLO-SAM/actions/workflows/ci.yaml/badge.svg)

---

## 🛠️ Recommended Conda Environment

```bash
conda create -n brain python=3.12
conda activate brain

# Install pip packages from requirements.txt
pip install -r requirements.txt
```

---

## 📂 Download Dataset

### 1️⃣ YOLO-Ready Object Detection Dataset

* URL: [brain-tumor.zip](https://github.com/ultralytics/assets/releases/download/v0.0.0/brain-tumor.zip)

### 2️⃣ Binary Classification / Simple Detection MRI Images

* URL: [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

Check folder structure with:

```bash
tree -d
```

---

## 🚀 API Endpoints / Features

### Endpoints

* **`/predict_vgg`**: Uses VGG16 model for classification, returns JSON with prediction (`tumor` / `no tumor`) and confidence.
* **`/detect_yolo`**: Uses YOLO for detection, returns annotated image with bounding boxes and confidence labels.
* **`/segment_sam`**: Uses SAM for segmentation, returns annotated image with masks.

### Image Processing

* All endpoints resize images to **512x512** for faster processing.

### Frontend Features

* **Three Action Buttons:**

  1. **Predict VGG** - Shows classification result with confidence bar
  2. **Detect YOLO** - Shows image with bounding boxes around detected tumors
  3. **Segment SAM** - Shows image with precise segmentation masks

* **Image Display:**

  * Uploaded image is displayed immediately.
  * Each prediction shows the processed/annotated result image.

### Key Features

* **Browse & Show Image:** Upload form displays the selected image instantly.
* **VGG16 Prediction:** Returns text result with confidence percentage.
* **YOLO Detection:** Returns image with green bounding boxes and confidence scores.
* **SAM Segmentation:** Returns image with colored segmentation masks for easy visualization.
* **Clean UI:** Updated result cards for each model type for better user experience.

---

## 📌 Notes

* The code is **modular and clean**, making it easy to extend with new models or endpoints.
* Frontend is designed to **help users understand results visually**.

---

## 🌱 Future Improvements

* Add **multi-class tumor classification**
* Integrate **more advanced models for detection**
* Add **batch image processing**
* Improve **frontend visualization and interactivity**

---

## 📬 Contact

**Author:** github.com/Ahmed2797

**Interest:** Deep Learning, Medical AI, Brain Tumor Detection
