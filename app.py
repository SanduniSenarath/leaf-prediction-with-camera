import os
import cv2
import numpy as np
import streamlit as st
import joblib
from skimage.feature import hog, local_binary_pattern

# ======================
# Load trained model
# ======================
MODEL_PATH = "best_leaf_model.pkl"

# Class labels and descriptions (English + Sinhala)
class_map = {
    0: {"name_en": "Bacterial Leaf Blight", 
        "name_si": "බැක්ටීරියා ලීෆ් බ්ලයිට්",
        "desc_en": "A bacterial disease causing yellowing and drying of leaf tips and margins.",
        "desc_si": "බැක්ටීරියාමගින් ඇති වන රෝගයක්, කොළ වල ඉගිලි සහ කෙළවරේ කොළ පෙරළි වීම හා වියළීම සිදු කරයි."},
    1: {"name_en": "Brown Spot", 
        "name_si": "බ්‍රවුන් ස්පොට්",
        "desc_en": "A fungal disease characterized by brown lesions on leaves.",
        "desc_si": "කොළ මත කළු කොළ පැහැති ලක්ෂණ ඇති කරන ෆංගල් රෝගයක්."},
    2: {"name_en": "Leaf Smut", 
        "name_si": "ලීෆ් ස්මට්",
        "desc_en": "A fungal disease that causes black spores on leaf surfaces.",
        "desc_si": "කොළ මත කළු ස්පෝර් ඇති කරන ෆංගල් රෝගයක්."}
}

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ======================
# Feature Extraction
# ======================
def extract_features(image):
    img = cv2.resize(image, (128, 128))

    # 1. Color Histogram
    hist = cv2.calcHist([img], [0,1,2], None, [10,10,10], [0,256,0,256,0,256]).flatten()

    # 2. HOG
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray, orientations=8, pixels_per_cell=(16,16),
                       cells_per_block=(1,1), visualize=False)

    # 3. LBP
    lbp = local_binary_pattern(gray, P=8, R=1)
    lbp_hist = np.histogram(lbp, bins=10)[0]

    # 4. HSV Histogram
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist([hsv], [0,1,2], None, [8,8,8], [0,180,0,256,0,256]).flatten()

    return np.concatenate([hist, hog_features, lbp_hist, hsv_hist])

# ======================
# Streamlit UI
# ======================
st.title("Rice Leaf Disease Detection / ගොයම් කොළ රෝග හඳුනා ගැනීම")
st.write("Take a photo or upload an image of a rice leaf to detect disease / කොළ රෝග හඳුනා ගැනීමට රූපයක් උඩුගත කරන්න")

# Language selection
language = st.radio("Select Language / භාෂාව තෝරන්න", ("English", "සිංහල"))

# Image input: camera or upload
choice = st.radio("Choose input method / රූප ලබාගැනීමේ ක්‍රමය තෝරන්න", ("Camera", "Upload / උඩුගත කරන්න"))

image = None
if choice == "Camera":
    camera_file = st.camera_input("Take a picture / රූපයක් ගන්න")
    if camera_file is not None:
        file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
elif choice == "Upload / උඩුගත කරන්න":
    uploaded_file = st.file_uploader("Upload Image / රූපය උඩුගත කරන්න", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

if image is not None:
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Input Image / ලබාදුන් රූපය", use_column_width=True)

    # Extract features and predict
    features = extract_features(image).reshape(1, -1)
    pred_class = model.predict(features)[0]

    # Display prediction
    if language == "English":
        st.subheader("Prediction Result")
        st.success(f"Disease Detected: **{class_map[pred_class]['name_en']}**")
        st.write(f"Description: {class_map[pred_class]['desc_en']}")
    else:
        st.subheader("ඵලය")
        st.success(f"හඳුනාගත් රෝගය: **{class_map[pred_class]['name_si']}**")
        st.write(f"විස්තරය: {class_map[pred_class]['desc_si']}")
