import numpy as np
from PIL import Image
import streamlit as st
import joblib
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
from matplotlib.colors import rgb_to_hsv

# ======================
# Load trained model
# ======================
MODEL_PATH = "best_leaf_model.pkl"

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
def extract_features(image: Image.Image):
    img = image.convert("RGB")
    img = np.array(img)
    img = np.array(Image.fromarray(img).resize((128,128)))

    # Color Histogram
    hist = np.histogramdd(img.reshape(-1,3), bins=(10,10,10), range=((0,255),(0,255),(0,255)))[0].flatten()

    # HOG
    gray = rgb2gray(img)
    hog_features = hog(gray, orientations=8, pixels_per_cell=(16,16), cells_per_block=(1,1), visualize=False)

    # LBP
    lbp = local_binary_pattern((gray*255).astype(np.uint8), P=8, R=1)
    lbp_hist = np.histogram(lbp, bins=10, range=(0,255))[0]

    # HSV
    hsv = rgb_to_hsv(img/255.0)
    hsv_hist = np.histogramdd(hsv.reshape(-1,3), bins=(8,8,8), range=((0,1),(0,1),(0,1)))[0].flatten()

    return np.concatenate([hist, hog_features, lbp_hist, hsv_hist])

# ======================
# Mobile-friendly CSS
# ======================
st.markdown("""
<style>
body {
    background-color: white;
}
h1, h2, h3, h4, h5, h6 {
    color: #006400; /* dark green */
}
.stButton>button {
    background-color: #FFA500;  /* orange buttons */
    color: white;
}
.stRadio>div, .stSelectbox>div {
    background-color: #FFFF66;  /* light yellow */
}
</style>
""", unsafe_allow_html=True)

# ======================
# Streamlit UI
# ======================
st.title("Rice Leaf Disease Detection / ගොයම් කොළ රෝග හඳුනා ගැනීම")

# Toggle for language
language = st.radio("Select Language / භාෂාව තෝරන්න", ["English", "සිංහල"])

# Image input method
choice = st.radio("Choose input / රූප ලබාගැනීමේ ක්‍රමය", ["Camera", "Upload / උඩුගත කරන්න"])

image = None
if choice == "Camera":
    camera_file = st.camera_input("Take a picture / රූපයක් ගන්න")
    if camera_file is not None:
        image = Image.open(camera_file)
elif choice == "Upload / උඩුගත කරන්න":
    uploaded_file = st.file_uploader("Upload Image / රූපය උඩුගත කරන්න", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

if image:
    st.image(image, caption="Input Image / ලබාදුන් රූපය", use_column_width=True)

    # Predict
    features = extract_features(image).reshape(1,-1)
    pred_class = model.predict(features)[0]

    if language == "English":
        st.subheader("Prediction Result")
        st.success(f"Disease Detected: **{class_map[pred_class]['name_en']}**")
        st.write(f"Description: {class_map[pred_class]['desc_en']}")
    else:
        st.subheader("ඵලය")
        st.success(f"හඳුනාගත් රෝගය: **{class_map[pred_class]['name_si']}**")
        st.write(f"විස්තරය: {class_map[pred_class]['desc_si']}")
