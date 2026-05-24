# 😷 Face Mask Detection System

A complete Deep Learning project that detects whether a person is wearing a face mask or not — in real time — using **MobileNetV2 (Transfer Learning)**, **OpenCV**, and a **Streamlit GUI**.

---

## 📁 Project Structure

```
face_mask_detection/
├── train.py                        ← Model training script
├── detect.py                       ← Real-time webcam / image detection (CLI)
├── app.py                          ← Streamlit interactive GUI
├── requirements.txt                ← Python package dependencies
├── README.md                       ← This file
│
├── mask_detector.keras             ← Generated after running train.py
├── accuracy_loss_plot.png          ← Generated after running train.py
│
├── face_detector/                  ← OpenCV DNN face detection files
│   ├── deploy.prototxt             ← Download link below
│   └── res10_300x300_ssd_iter_140000.caffemodel
│
└── dataset/                        ← Your training data
    ├── with_mask/                  ← Images of people WITH masks
    │   ├── image1.jpg
    │   └── ...
    └── without_mask/               ← Images of people WITHOUT masks
        ├── image1.jpg
        └── ...
```

---

## 🚀 Setup Instructions

### Step 1 — Clone / Download the project

```bash
git clone <your-repo-url>
cd face_mask_detection
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Download Required Files

### A) Dataset

Download the **Face Mask Detection** dataset from Kaggle:

> 🔗 https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

Extract and place images like this:
```
dataset/
├── with_mask/      ← ~3725 images
└── without_mask/   ← ~3828 images
```

---

### B) OpenCV Face Detector Files

Download two files and place them in a `face_detector/` folder:

**File 1 — deploy.prototxt**
```
https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
```

**File 2 — res10_300x300_ssd_iter_140000.caffemodel**
```
https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
```

Or use these wget commands:
```bash
mkdir face_detector
cd face_detector

wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt

wget https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

cd ..
```

---

## 🏋️ Step 4 — Train the Model

```bash
python train.py
```

This will:
- Load images from `dataset/with_mask/` and `dataset/without_mask/`
- Preprocess and augment the data
- Build a MobileNetV2 transfer learning model
- Train for 10 epochs
- Save `mask_detector.keras`
- Save `accuracy_loss_plot.png`
- Print classification report

**Expected output:**
```
[INFO] Loading images from dataset...
[INFO] Total images loaded: 7553
[INFO] Training samples : 6042
[INFO] Testing  samples : 1511
[INFO] Building model with MobileNetV2...
[INFO] Training head layers...
Epoch 1/10 ...
...
[INFO] Saving model to: mask_detector.keras
[INFO] Plot saved to: accuracy_loss_plot.png
```

**Expected accuracy:** ~96–98% validation accuracy

---

## 🖥️ Step 5 — Run the GUI

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

### Features:
- **Image Upload tab** — Upload any photo, get annotated result + per-face confidence
- **Webcam tab** — Capture frames via browser webcam
- **About tab** — Project details, architecture diagram, quick-start guide

---

## 📷 Step 6 — Real-Time Webcam (Command Line)

```bash
# Webcam detection (press Q to quit)
python detect.py

# Single image detection
python detect.py --image /path/to/photo.jpg
```

---

## 📊 Model Architecture

```
Input (224×224×3)
        ↓
MobileNetV2 (pretrained ImageNet, frozen)
        ↓
AveragePooling2D(7×7)
        ↓
Flatten
        ↓
Dense(128, activation='relu')
        ↓
Dropout(0.5)
        ↓
Dense(2, activation='softmax')
        ↓
Output: [P(Mask), P(No Mask)]
```

---

## 📈 Expected Results

| Metric | Expected Value |
|--------|---------------|
| Training Accuracy | ~98% |
| Validation Accuracy | ~96% |
| Precision (Mask) | ~97% |
| Recall (Mask) | ~96% |
| F1 Score | ~96% |

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `mask_detector.keras not found` | Run `python train.py` first |
| `Face detector files not found` | Download the two `.prototxt`/`.caffemodel` files |
| `Could not open webcam` | Check webcam connection; try changing `VideoCapture(0)` to `VideoCapture(1)` |
| Low accuracy | Ensure dataset is clean; increase epochs in `train.py` |
| Slow on CPU | Normal — MobileNetV2 is optimized but still needs a few seconds per batch |

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.8+ | Core language |
| TensorFlow / Keras | Model building & training |
| MobileNetV2 | Pretrained CNN backbone |
| OpenCV | Face detection, image I/O, webcam |
| NumPy | Array operations |
| Scikit-learn | Train/test split, metrics |
| Matplotlib | Training plots |
| Streamlit | Interactive web GUI |
| Pillow | Image format handling |

---

## 👨‍🎓 College Project Notes

- **Viva tip:** Be ready to explain Transfer Learning — why we freeze base layers and only train the custom head.
- **Demo flow:** Show `streamlit run app.py` → upload a test image → webcam tab → show the accuracy/loss plot.
- **Report:** Include the architecture diagram, training curves, classification report, and a few sample predictions.

---

*Built for educational purposes as a college Deep Learning project.*
