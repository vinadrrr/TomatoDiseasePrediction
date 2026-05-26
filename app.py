import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# 1. BROWSER TAB CONFIGURATION
st.set_page_config(page_title="Tomato Leaf Doctor", page_icon="🍅", layout="centered")

# 2. CUSTOM VISUAL THEME (Tomato Red Accents)
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

# --- TEMP DIAGNOSTIC MODEL LOADER ---
@st.cache_resource
def load_tomato_model():
    return tf.keras.models.load_model('tomato_model.h5')

# Print out files in the server directory to see why it's hiding
current_directory_files = os.listdir('.')
st.write("### 🔍 Server Directory Scan:")
st.write(current_directory_files)

try:
    model = load_tomato_model()
    st.success("🎯 Success! Model found and loaded smoothly.")
except Exception as e:
    st.error(f"❌ Error Loading Model: {e}")
    
# 4. CLASS NAME DEFINITIONS
# Ensure these match the exact folder names from your dataset in alphabetical order!
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

# 5. USER INTERFACE: FILE UPLOADER
uploaded_file = st.file_uploader("Upload leaf image (JPG/PNG)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image to the user
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Tomato Leaf', use_column_width=True)
    
    # 6. INFERENCE PIPELINE RUNNER
    if st.button('Run Diagnostics'):
        with st.spinner('Analyzing plant tissue anomalies...'):
            # Image Preprocessing (Must match your training input parameters)
            img = image.resize((224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) # Add batch dimension (1, 224, 224, 3)
            img_array = img_array / 255.0 # Normalize pixel vectors to 0-1 range
            
            # Execute model prediction
            predictions = model.predict(img_array)
            highest_score_index = np.argmax(predictions[0])
            
            # Extract the highest probability value directly
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
