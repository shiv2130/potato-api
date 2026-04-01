# 🥔 Potato Disease Detection API

A lightweight REST API that powers the [Plant Disease Detector](https://github.com/shiv2130/App_for_Plant_Disease_Detector) Android app. It accepts plant leaf images and returns a predicted disease class using a trained deep learning model.

---

## 📌 Overview

This backend is built with **FastAPI** and serves an **ONNX** model trained on the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease). It currently supports classification of **potato plant diseases** into three categories:

- ✅ Healthy
- 🦠 Early Blight
- 🦠 Late Blight

The Android app sends an image to this API and displays the prediction result to the user.

---

## ✨ Features

- 📤 **Image Upload Endpoint** — accepts `.jpg`/`.png` leaf images via `multipart/form-data`
- 🤖 **ML Inference** — runs a trained CNN model and returns the predicted class + confidence score
- ⚡ **Fast & Async** — built on FastAPI for high-performance async request handling
- 🔁 **JSON Response** — clean, structured output ready for mobile consumption

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Framework | FastAPI |
| Language | Python 3.9+ |
| ML Runtime | ONNX Runtime |
| Model Format | `.onnx` |
| Server | Uvicorn |
| Image Processing | Pillow / NumPy |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- `pip` package manager
- A trained model file (`.onnx`)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/shiv2130/potato-api.git
   cd potato-api
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your trained model**

   Place your `.onnx` model file in the project directory and update the model path in `main.py`:
   ```python
   import onnxruntime as ort
   session = ort.InferenceSession("./models/potato_model.onnx")
   ```

5. **Run the server**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

   The API will be live at `http://localhost:8000`

---

## 📡 API Reference

### `GET /`
Health check endpoint.

**Response:**
```json
{ "message": "Plant Disease Detection API is running." }
```

---

### `POST /predict`
Upload a plant leaf image to get a disease prediction.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` — the image file (`.jpg` or `.png`)

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@leaf.jpg"
```

**Response:**
```json
{
  "class": "Potato Early Blight",
  "confidence": 0.97
}
```

---

## 📁 Project Structure

```
potato-api/
├── main.py               # FastAPI app and prediction endpoint
├── requirements.txt      # Python dependencies
├── models/
│   └── potato_model.onnx # Trained ONNX model (add your own)
└── README.md
```

---

## 🧠 Model Details

The model is a Convolutional Neural Network (CNN) exported to **ONNX format** and served via **ONNX Runtime** for fast, framework-agnostic inference. It was trained on the **PlantVillage** dataset and classifies potato leaf images into one of three categories:

| Class | Label Index |
|---|---|
| Potato Early Blight | 0 |
| Potato Late Blight | 1 |
| Potato Healthy | 2 |

Input images are resized to **256×256** and normalized before inference.

---

## 📦 Deployment

This API can be deployed to any cloud platform that supports Python. Common options:

- **Google Cloud Run** — containerize with Docker and deploy serverlessly
- **AWS EC2 / Elastic Beanstalk**
- **Railway / Render** — simple push-to-deploy from GitHub
- **Heroku**

### Docker (optional)

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install fastapi uvicorn onnxruntime pillow numpy
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t potato-api .
docker run -p 8000:8000 potato-api
```

---

## 🔗 Related

- 📱 **Android App** → [App_for_Plant_Disease_Detector](https://github.com/shiv2130/App_for_Plant_Disease_Detector)

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for bug fixes, model improvements, or new plant/disease support.

---

## 👤 Author

**Shiv** — [@shiv2130](https://github.com/shiv2130)

---

> 🌿 *Part of a full-stack AI solution for accessible plant disease diagnosis.*
