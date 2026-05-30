import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# BROWSER TAB CONFIGURATION
st.set_page_config(page_title="Tomato Leaf Doctor", page_icon="🍅", layout="centered")

# VISUAL THEME 
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

# CACHED MODEL LOADER 
@st.cache_resource
def load_tomato_model():
    return tf.keras.models.load_model('tomato_model.h5')

try:
    model = load_tomato_model()
except Exception as e:
    st.error("Could not find 'tomato_model.h5'. Ensure it is in the same folder as this script!")

# CLASS NAME DEFINITIONS
CLASS_NAMES = [
    'Bacterial_spot',               
    'Early_blight',                  
    'Late_blight',               
    'Leaf_Mold',                
    'Septoria_leaf_spot',        
    'Spider_mites Two-spotted_spider_mite',
    'Target_Spot',                
    'Tomato_mosaic_virus',          
    'Tomato_Yellow_Leaf_Curl_Virus', 
    'healthy',                     
    'powdery_mildew'                  
]

# FILE UPLOADER
uploaded_file = st.file_uploader("Upload leaf image (JPG/PNG)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Tomato Leaf', use_column_width=True)
    
    if st.button('Run Diagnostics'):
        with st.spinner('Analyzing plant tissue anomalies...'):
            # Image Preprocessing 
            img = image.resize((224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) 
            img_array = img_array / 255.0 
            
            # Execute model prediction
            predictions = model.predict(img_array)
            highest_score_index = np.argmax(predictions[0])
            
            confidence = predictions[0][highest_score_index] * 100
        
            prediction_label = CLASS_NAMES[highest_score_index]
            
            # Render Diagnostic Report Output UI
            st.write("---")
            st.subheader("Diagnostic Report:")
            
            if "Healthy" in prediction_label:
                st.success(f"**Result:** {prediction_label} ({confidence:.1f}% Confidence)")
                st.balloons()
            else:
                st.error(f"**Detected Condition:** {prediction_label} ({confidence:.1f}% Confidence)")
                st.warning("Action Recommended: Check isolation protocols and consult a targeted crop treatment guide.")