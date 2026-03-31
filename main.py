# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf
# import math

# app = FastAPI()

# #  CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# #  Load model
# try:
#     MODEL = tf.keras.models.load_model(
#         r"C:\Users\singh\OneDrive\Desktop\plant detection system\api\Plant_disease_model.h5"
#     )
#     print(" Model loaded successfully")
#     print(" Model input shape:", MODEL.input_shape)
# except Exception as e:
#     print(f" Failed to load model: {e}")
#     MODEL = None


# CLASS_NAMES = ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy", "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy", "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot", "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus", "Tomato_healthy"]


# def get_target_size():
#     """Automatically derive image resize target and whether to flatten from model input shape."""
#     input_shape = MODEL.input_shape  # e.g. (None, 224, 224, 3) or (None, 86528)

#     if len(input_shape) == 4:
#         # Normal CNN — expects (batch, H, W, C)
#         return (input_shape[1], input_shape[2]), False
#     elif len(input_shape) == 2:
#         # Dense/flat model — expects (batch, H*W*C)
#         flat_size = input_shape[1]
#         channels = 3
#         hw = int(math.sqrt(flat_size / channels))
#         return (hw, hw), True
#     else:
#         return (224, 224), False


# # Health check — shows model input shape so you can verify
# @app.get("/ping")
# async def ping():
#     return {
#         "message": "Server is running ",
#         "model_loaded": MODEL is not None,
#         "model_input_shape": str(MODEL.input_shape) if MODEL else "N/A",
#     }


# #  Image preprocessing — auto-adapts to whatever shape model expects
# def read_file_as_image(data: bytes) -> np.ndarray:
#     target_size, is_flat = get_target_size()

#     image = Image.open(BytesIO(data)).convert("RGB")
#     image = image.resize((target_size[1], target_size[0]))  # PIL takes (
#     image = np.array(image, dtype=np.float32)  # values between 0-255

#     if is_flat:
#         image = image.flatten()  # shape becomes (H*W*C,)

#     return image


# #  Prediction endpoint
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     if MODEL is None:
#         raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

#     if not file.content_type.startswith("image/"):
#         raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

#     try:
#         contents = await file.read()
#         image = read_file_as_image(contents)
#         img_batch = np.expand_dims(image, 0)  # add batch dimension

#         print(" Batch shape sent to model:", img_batch.shape)

#         predictions = MODEL.predict(img_batch)
#         predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#         confidence = float(np.max(predictions[0]))

#         return {
#             "class": predicted_class,
#             "confidence": confidence,
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# #  Run server
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
import os
import uvicorn
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ONNX Model Loading ---
MODEL_PATH = "Plant_disease_model.onnx"

try:
    # Create an Inference Session
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    
    # Get model metadata
    input_details = session.get_inputs()[0]
    input_name = input_details.name
    input_shape = input_details.shape  # e.g., [None, 224, 224, 3] or [None, 150528]
    
    print(f"✅ ONNX Model loaded successfully")
    print(f"Input Name: {input_name} | Input Shape: {input_shape}")
except Exception as e:
    print(f"❌ Failed to load ONNX model: {e}")
    session = None

CLASS_NAMES = ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy", "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy", "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot", "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus", "Tomato_healthy"]


def get_target_params():
    """Derive resize dimensions and flattening from ONNX input shape."""
    # input_shape is usually [Batch, Height, Width, Channels] or [Batch, FlatSize]
    if len(input_shape) == 4:
        return (input_shape[1], input_shape[2]), False
    elif len(input_shape) == 2:
        # For flattened models (Dense), assuming square RGB image
        flat_size = input_shape[1]
        hw = int((flat_size / 3) ** 0.5)
        return (hw, hw), True
    return (224, 224), False

def read_file_as_image(data: bytes) -> np.ndarray:
    target_size, is_flat = get_target_params()
    
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((target_size[1], target_size[0])) 
    
    # Preprocessing: Convert to float32 and normalize if your model requires it (e.g., / 255.0)
    # Most ONNX models exported from Keras expect the same scaling as the original model
    img_array = np.array(image, dtype=np.float32) 
    
    # Normalize if necessary (Uncomment the line below if your model expects 0-1 range)
    # img_array = img_array / 255.0

    if is_flat:
        img_array = img_array.flatten()

    return img_array

@app.get("/ping")
async def ping():
    return {
        "message": "Server is running",
        "model_loaded": session is not None,
        "input_shape": input_shape
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        image = read_file_as_image(contents)
        
        # ONNX requires explicit batch dimension
        img_batch = np.expand_dims(image, axis=0)

        # Run Inference
        # session.run(output_names, input_feed_dict)
        outputs = session.run(None, {input_name: img_batch})
        
        predictions = outputs[0]
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return {
            "class": predicted_class,
            "confidence": confidence,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)