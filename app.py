import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import os
import urllib.request

# 1. BROWSER TAB CONFIGURATION
st.set_page_config(page_title="Tomato Leaf Doctor", page_icon="🍅", layout="centered")

# Custom UI styling
st.markdown("""
<style>
    .stButton>button {
        color: white;
        background-color: #d32f2f; 
        border-radius: 10px;
        font-weight: bold;
    }
    h1 {
        color: #b71c1c;
    }
</style>
""", unsafe_allow_html=True)

st.title("🍅 Tomato Leaf Disease Analyzer")
st.write("Upload a clear photo of a single tomato leaf to detect potential diseases.")

# --- 2. AUTOMATIC MODEL DOWNLOAD PIPELINE ---
MODEL_PATH = "tomato_model.h5"

# 🛑 REPLACE THIS URL STRING WITH YOUR GOOGLE DRIVE OR DROPBOX LINK
# If using Google Drive, make sure it is a direct download link or paste your raw sharing link here!
SHARED_LINK = "https://drive.google.com/file/d/1ouodYp5pGKjXVM9d6hGU_3Fy1qcdZ_cj/view?usp=sharing"

@st.cache_resource
def load_tomato_model():
    # If the model file isn't in the folder, download it directly from your cloud link
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Initializing server weights (this happens once on startup)..."):
            try:
                # Helper function to handle Google Drive conversion to direct download link
                url = SHARED_LINK
                if "drive.google.com" in url and "file/d/" in url:
                    file_id = url.split("file/d/")[1].split("/")[0]
                    url = f"https://docs.google.com/uc?export=download&id={file_id}"
                elif "dropbox.com" in url:
                    url = url.replace("?dl=0", "?dl=1")
                
                urllib.request.urlretrieve(url, MODEL_PATH)
            except Exception as download_error:
                st.error(f"Failed to fetch model from cloud storage: {download_error}")
                
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_tomato_model()
except Exception as e:
    st.error(f"❌ Error initializing network framework: {e}")

# 3. CLASS NAME DEFINITIONS
CLASS_NAMES = [
    'Bacterial Spot', 
    'Early Blight', 
    'Late Blight', 
    'Leaf Mold', 
    'Septoria Leaf Spot', 
    'Spider Mites', 
    'Target Spot', 
    'Yellow Leaf Curl Virus', 
    'Mosaic Virus', 
    'Healthy'
]

# 4. USER INTERFACE: FILE UPLOADER
uploaded_file = st.file_uploader("Upload leaf image (JPG/PNG)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Tomato Leaf', use_column_width=True)
    
    if st.button('Run Diagnostics'):
        with st.spinner('Analyzing plant tissue anomalies...'):
            img = image.resize((224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            
            predictions = model.predict(img_array)
            highest_score_index = np.argmax(predictions[0])
            confidence = predictions[0][highest_score_index] * 100
            
            prediction_label = CLASS_NAMES[highest_score_index]
            
            st.write("---")
            st.subheader("Diagnostic Report:")
            
            if "Healthy" in prediction_label:
                st.success(f"**Result:** {prediction_label} ({confidence:.1f}% Confidence)")
                st.balloons()
            else:
                st.error(f"**Detected Condition:** {prediction_label} ({confidence:.1f}% Confidence)")
                st.warning("Action Recommended: Check isolation protocols and consult a targeted crop treatment guide.")