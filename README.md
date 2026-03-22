# 🧠 Fetal Ultrasound Abdomen Segmentation using U-Net

## 📌 Overview

This project focuses on **automatic segmentation of fetal abdomen regions** from ultrasound images using a deep learning approach based on the U-Net architecture. The goal is to assist in **accurate biometric analysis** and improve accessibility to fetal health assessment, especially in low-resource settings.

---

## 🚀 Features

* ✅ U-Net based deep learning model for medical image segmentation
* ✅ Handles grayscale ultrasound images
* ✅ End-to-end training pipeline
* ✅ Saves trained model (`.pth`) for inference
* ✅ Designed for fetal abdomen region extraction

---

## 🏗️ Project Structure

```
├── dataset.py        # Data loading and preprocessing
├── unet.py           # U-Net model architecture
├── train.py          # Training pipeline
├── data/             # Dataset (images & masks)
├── models/           # Saved models (.pth)
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/FetalUltrasound-AbdomenSegmentation.git
cd FetalUltrasound-AbdomenSegmentation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📂 Dataset

* The dataset should contain:

  * **Ultrasound images**
  * **Corresponding segmentation masks**
* Place them inside the `data/` directory.

Example structure:

```
data/
├── images/
├── masks/
```

---

## 🏋️ Training the Model

Run the training script:

```bash
python train.py
```

After training:

* The model will be saved as:

```
models/unet.pth
```

---

## 🧪 Inference (Prediction)

You can load the trained model and run predictions:

```python
import torch
from unet import UNet

model = UNet()
model.load_state_dict(torch.load("models/unet.pth"))
model.eval()
```

---

## 📊 Applications

* 📏 Fetal abdominal circumference estimation
* 🩺 Prenatal health monitoring
* 🤖 AI-assisted ultrasound systems
* 🌍 Rural healthcare support tools

---

## 🧠 Model Details

* Architecture: **U-Net**
* Input: Ultrasound images
* Output: Segmented abdomen mask
* Framework: PyTorch

---

## 🔮 Future Improvements

* 🔹 Add Dice Loss & IoU metrics
* 🔹 Improve model with Attention U-Net
* 🔹 Real-time inference integration
* 🔹 Deployment with Streamlit / Web App
* 🔹 Integration with robotic ultrasound systems

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

Developed by **[VASU P]**
AI/ML Engineer | Medical Imaging Enthusiast
