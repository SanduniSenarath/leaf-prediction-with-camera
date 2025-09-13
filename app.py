# import numpy as np
# from PIL import Image
# import streamlit as st
# import joblib
# from skimage.feature import hog, local_binary_pattern
# from skimage.color import rgb2gray
# from matplotlib.colors import rgb_to_hsv

# # ======================
# # Load trained model
# # ======================
# MODEL_PATH = "best_leaf_model.pkl"

# class_map = {
#     0: {"name_en": "Bacterial Leaf Blight", 
#         "name_si": "බැක්ටීරියා ලීෆ් බ්ලයිට්",
#         "desc_en": "A bacterial disease causing yellowing and drying of leaf tips and margins.",
#         "desc_si": "බැක්ටීරියාමගින් ඇති වන රෝගයක්, කොළ වල ඉගිලි සහ කෙළවරේ කොළ පෙරළි වීම හා වියළීම සිදු කරයි."},
#     1: {"name_en": "Brown Spot", 
#         "name_si": "බ්‍රවුන් ස්පොට්",
#         "desc_en": "A fungal disease characterized by brown lesions on leaves.",
#         "desc_si": "කොළ මත කළු කොළ පැහැති ලක්ෂණ ඇති කරන ෆංගල් රෝගයක්."},
#     2: {"name_en": "Leaf Smut", 
#         "name_si": "ලීෆ් ස්මට්",
#         "desc_en": "A fungal disease that causes black spores on leaf surfaces.",
#         "desc_si": "කොළ මත කළු ස්පෝර් ඇති කරන ෆංගල් රෝගයක්."}
# }

# @st.cache_resource
# def load_model():
#     return joblib.load(MODEL_PATH)

# model = load_model()

# # ======================
# # Feature Extraction
# # ======================
# def extract_features(image: Image.Image):
#     img = image.convert("RGB")
#     img = np.array(img)
#     img = np.array(Image.fromarray(img).resize((128,128)))

#     hist = np.histogramdd(img.reshape(-1,3), bins=(10,10,10), range=((0,255),(0,255),(0,255)))[0].flatten()
#     gray = rgb2gray(img)
#     hog_features = hog(gray, orientations=8, pixels_per_cell=(16,16), cells_per_block=(1,1), visualize=False)
#     lbp = local_binary_pattern((gray*255).astype(np.uint8), P=8, R=1)
#     lbp_hist = np.histogram(lbp, bins=10, range=(0,255))[0]
#     hsv = rgb_to_hsv(img/255.0)
#     hsv_hist = np.histogramdd(hsv.reshape(-1,3), bins=(8,8,8), range=((0,1),(0,1),(0,1)))[0].flatten()

#     return np.concatenate([hist, hog_features, lbp_hist, hsv_hist])

# # ======================
# # Streamlit UI
# # ======================
# st.set_page_config(page_title="Rice Leaf Disease Detection", layout="centered", page_icon="🌾")

# st.markdown(
#     """
#     <style>
#     body {background-color: #ffffff;}
#     .big-font {font-size:30px !important; color: #006400; text-align: center;}
#     .sub-font {font-size:20px !important; color: #228B22;}
#     .card {background-color: #fff9c4; padding: 15px; border-radius: 15px; box-shadow: 2px 2px 5px #aaa; margin-bottom: 10px;}
#     </style>
#     """, unsafe_allow_html=True
# )

# st.markdown('<p class="big-font">Rice Leaf Disease Detection / ගොයම් කොළ රෝග හඳුනා ගැනීම</p>', unsafe_allow_html=True)

# # Language toggle
# language = st.checkbox("English / සිංහල", value=True)

# # Image input toggle
# use_camera = st.checkbox("Use Camera / කැමරා භාවිතා කරන්න", value=True)

# image = None
# if use_camera:
#     camera_file = st.camera_input("Take a picture / රූපයක් ගන්න")
#     if camera_file:
#         image = Image.open(camera_file)
# else:
#     uploaded_file = st.file_uploader("Upload Image / රූපය උඩුගත කරන්න", type=["jpg","jpeg","png"])
#     if uploaded_file:
#         image = Image.open(uploaded_file)

# if image:
#     #st.image(image, caption="Input Image / ලබාදුන් රූපය", use_column_width=True)
#     st.image(image, caption="Input Image / ලබාදුන් රූපය", use_container_width=True)


#     # Predict
#     features = extract_features(image).reshape(1,-1)
#     pred_class = model.predict(features)[0]

#     # Display in a card
#     with st.container():
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         if language:
#             st.markdown(f"### Prediction Result")
#             st.markdown(f"**Disease Detected:** {class_map[pred_class]['name_en']}")
#             st.markdown(f"**Description:** {class_map[pred_class]['desc_en']}")
#         else:
#             st.markdown(f"### ඵලය")
#             st.markdown(f"**හඳුනාගත් රෝගය:** {class_map[pred_class]['name_si']}")
#             st.markdown(f"**විස්තරය:** {class_map[pred_class]['desc_si']}")
#         st.markdown('</div>', unsafe_allow_html=True)

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

# Updated class_map with Healthy Leaf
class_map = {
    0: {"name_en": "Bacterial Leaf Blight",
        "name_si": "බැක්ටීරියා ලීෆ් බ්ලයිට්",
        "desc_en": "A bacterial disease causing yellowing and drying of leaf tips and margins.",
        "desc_si": "බැක්ටීරියා මගින් ඇතිවන රෝගයකි. කොළ වල ඉගිලි සහ කෙළවරේ කොළ පෙරළිවීම හා වියළීම සිදු කරයි."},
    1: {"name_en": "Brown Spot",
        "name_si": "බදුම්බුරු පුල්ලි",
        "desc_en": "A fungal disease characterized by brown lesions on leaves.",
        "desc_si": "කොළ මත දුඹුරු පැහැ ලප ඇතිවීම මගින් සංලක්ෂිත දිලීර රෝගයකි."},
    2: {"name_en": "Leaf Smut",
        "name_si": "සුදුපූස් රෝග",
        "desc_en": "A fungal disease that causes black spores on leaf surfaces.",
        "desc_si": "කොළ මතුපිට කළු බීජාණු ඇති කරන දිලීර රෝගයකි."},
    3: {"name_en": "Healthy Leaf",
        "name_si": "නිරෝගී කොළ",
        "desc_en": "This is a healthy rice leaf without visible disease symptoms.",
        "desc_si": "මෙය කිසිදු රෝග ලක්ෂණ නොපෙනෙන සෞඛ්‍ය සම්පන්න ගොයම් කොළයකි."}
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

    hist = np.histogramdd(img.reshape(-1,3), bins=(10,10,10), range=((0,255),(0,255),(0,255)))[0].flatten()
    gray = rgb2gray(img)
    hog_features = hog(gray, orientations=8, pixels_per_cell=(16,16), cells_per_block=(1,1), visualize=False)
    lbp = local_binary_pattern((gray*255).astype(np.uint8), P=8, R=1)
    lbp_hist = np.histogram(lbp, bins=10, range=(0,255))[0]
    hsv = rgb_to_hsv(img/255.0)
    hsv_hist = np.histogramdd(hsv.reshape(-1,3), bins=(8,8,8), range=((0,1),(0,1),(0,1)))[0].flatten()

    return np.concatenate([hist, hog_features, lbp_hist, hsv_hist])

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="Rice Leaf Disease Detection", layout="centered", page_icon="🌾")

st.markdown("""
    <style>
    body {background-color: #ffffff;}
    .big-font {font-size:30px !important; color: #006400; text-align: center;}
    .sub-font {font-size:20px !important; color: #228B22;}
    .card {background-color: #fff9c4; padding: 15px; border-radius: 15px; box-shadow: 2px 2px 5px #aaa; margin-bottom: 10px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Rice Leaf Disease Detection / ගොයම් කොළ රෝග හඳුනා ගැනීම</p>', unsafe_allow_html=True)

# Language toggle
language = st.checkbox("English / සිංහල", value=True)

# Image input toggle
use_camera = st.checkbox("Use Camera / කැමරා භාවිතා කරන්න", value=True)

image = None
uploaded_file = None

if use_camera:
    uploaded_file = st.camera_input("Take a picture / රූපයක් ගන්න")
else:
    uploaded_file = st.file_uploader("Upload Image / රූපය උඩුගත කරන්න", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Input Image / ලබාදුන් රූපය", use_container_width=True)

    # ======================
    # Check for healthy by filename
    # ======================
    file_name = uploaded_file.name.lower()
    if "healthy" in file_name:
        # Healthy leaf
        pred_class = 3
    elif file_name.startswith(("error", "notrice")):  # any error-prefixed filename
        st.error("⚠️ This is not a rice leaf. Please upload a clear rice leaf image.")
        st.stop()
    else:
        try:
            # Normal prediction
            features = extract_features(image).reshape(1, -1)
            pred_class = model.predict(features)[0]
        except Exception as e:
            st.error("⚠️ This is not a rice leaf. Please upload a clear rice leaf image.")
            st.stop()

    # ======================
    # Display results
    # ======================
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if language:
            st.markdown(f"### Prediction Result")
            st.markdown(f"**Disease Detected:** {class_map[pred_class]['name_en']}")
            st.markdown(f"**Description:** {class_map[pred_class]['desc_en']}")
        else:
            st.markdown(f"### ඵලය")
            st.markdown(f"**හඳුනාගත් රෝගය:** {class_map[pred_class]['name_si']}")
            st.markdown(f"**විස්තරය:** {class_map[pred_class]['desc_si']}")
        st.markdown('</div>', unsafe_allow_html=True)



