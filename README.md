# 🚗 Real-Time Automatic License Plate Recognition (ALPR) System

A robust, real-time end-to-end Computer Vision pipeline for detecting, cropping, reading, and tracking vehicle license plates in video streams and live camera feeds using **YOLOv8** and **EasyOCR**.

---

## 📌 Features

- **High-Accuracy License Plate Localization:** Powered by a fine-tuned YOLOv8 model (`license_plate_best.pt`) optimized for diverse lighting and perspective variations.
- **Image Preprocessing for Enhanced OCR:** Crops the detected plate and applies Grayscale conversion, Otsu thresholding, and bicubic upscaling to maximize character legibility.
- **OCR with Domain-Specific Post-Processing:** Uses EasyOCR with character whitelist and heuristic mapping (e.g., resolving `0` vs `O`, `1` vs `I`, `8` vs `B`) to correct common optical character recognition errors.
- **Temporal Multi-Frame Smoothing:** Uses spatial pseudo-tracking and deque majority voting to prevent flickering and stabilize OCR reads across consecutive video frames.
- **Modular & Deployable:** Provides both a production-ready CLI script (`main.py`) and an exploratory notebook (`number_plate_detection.ipynb`).

---

## 🏗️ Architecture Pipeline

```text
[ Input Frame (Video / Webcam) ]
              │
              ▼
    [ YOLOv8 Object Detector ]  ──► (Bounding Box Localization)
              │
              ▼
   [ Plate Crop & Preprocessing ]
   - Grayscale Conversion
   - Otsu's Binary Thresholding
   - Bicubic Upscaling (2x)
              │
              ▼
    [ EasyOCR Text Extraction ] ──► (Raw Characters)
              │
              ▼
   [ Error Correction & Regex ] ──► (Format Disambiguation)
              │
              ▼
   [ Temporal Majority Voting ] ──► (Stabilized Output Plate)
              │
              ▼
[ Annotated Frame / Output Video ]
```

---

## 🛠️ Tech Stack & Requirements

- **Deep Learning / Object Detection:** `ultralytics` (YOLOv8)
- **Optical Character Recognition (OCR):** `easyocr` (PyTorch backend)
- **Computer Vision & Image Processing:** `opencv-python`, `numpy`
- **Pattern Matching:** Python Standard Library `re`, `collections`

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/automatic-license-plate-recognition.git
cd automatic-license-plate-recognition
```

### 2. Set Up a Virtual Environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Run on Webcam Feed:
```bash
python main.py --input 0
```

### Run on a Video File:
```bash
python main.py --input path/to/video.mp4
```

### Run and Save Processed Output:
```bash
python main.py --input path/to/video.mp4 --output output.mp4
```

### Optional Arguments:
- `--conf`: Set detection confidence threshold (default: `0.3`).
- `--model`: Path to model weights (default: `license_plate_best.pt`).
- `--gpu`: Enable GPU acceleration for EasyOCR (default: `True`).

---

## 📁 Repository Structure

```text
automatic-license-plate-recognition/
├── license_plate_best.pt        # Trained YOLOv8 weights for license plate detection
├── main.py                      # Main production inference and tracking script
├── number_plate_detection.ipynb # Research & exploratory notebook
├── requirements.txt             # Python dependencies
├── .gitignore                   # Excludes heavy videos and virtual environments
└── README.md                    # Project documentation
```

---

## 📄 License
This project is open-source under the MIT License.
