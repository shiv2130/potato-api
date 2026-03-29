from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import io
import onnxruntime as ort

app = FastAPI()

# Load ONNX model
session = ort.InferenceSession("model.onnx")

# Get input name
input_name = session.get_inputs()[0].name

class_names = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
               "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
               "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
               "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
               "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus",
               "Tomato_healthy"]

def read_file_as_image(data):
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return image.astype(np.float32)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    outputs = session.run(None, {input_name: img_batch})
    predictions = outputs[0]

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {
        "class": predicted_class,
        "confidence": confidence
    }