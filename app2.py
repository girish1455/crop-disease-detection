import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import base64
import json

# ============ Load Model & Classes ============
model = load_model("crop_classifier_model.h5")

with open("class_names.json", "r") as f:
    class_indices = json.load(f)
index_to_label = {v: k for k, v in class_indices.items()}

# ============ Disease Info ============
disease_info = {
    "Bacterial Blight": {
        "Symptoms": "Leaves show water-soaked streaks that turn yellow and dry out, often starting at the leaf tip. Scalding and wilting may occur under severe infections.",
        "Cure": "Remove and destroy infected plants to prevent spread. Apply copper-based bactericides early in the disease cycle for effective control.",
        "Prevention": "Ensure good drainage, avoid overhead irrigation, use certified disease-free seeds, and avoid overcrowding of plants."
    },
    "Corn___Common_Rust": {
        "Symptoms": "Formation of small, circular to oval, reddish-brown pustules mostly on upper leaf surfaces. Severe cases lead to leaf blight and reduced photosynthesis.",
        "Cure": "Grow rust-resistant corn hybrids. Apply fungicides such as strobilurins or triazoles when rust is first detected.",
        "Prevention": "Plant at recommended times, manage irrigation to reduce humidity, and maintain proper row spacing to improve air circulation."
    },
    "Corn___Gray_Leaf_Spot": {
        "Symptoms": "Development of elongated, rectangular, gray to tan lesions on the leaf surface, often bordered by dark margins. Severe infections cause premature leaf death.",
        "Cure": "Timely application of fungicides like Trifloxystrobin or Azoxystrobin can help control the spread.",
        "Prevention": "Use crop rotation, remove infected plant residues post-harvest, and choose tolerant corn varieties."
    },
    "Corn___Healthy": {
        "Symptoms": "Leaves are green and firm with no visible lesions, deformities, or signs of stress. Plant grows uniformly with consistent development.",
        "Cure": "No cure needed for healthy plants.",
        "Prevention": "Implement integrated pest management, maintain field hygiene, and regularly monitor for early signs of disease."
    },
    "Corn___Northern_Leaf_Blight": {
        "Symptoms": "Appearance of long, narrow, cigar-shaped lesions that are gray-green and become tan as they mature. Can lead to substantial yield loss.",
        "Cure": "Apply foliar fungicides such as Pyraclostrobin if the infection level is high, especially during tasseling to silking stage.",
        "Prevention": "Use resistant hybrids, rotate crops with non-hosts, and plow under crop debris to reduce overwintering spores."
    },
    "Healthy": {
        "Symptoms": "Plant shows robust growth, no discoloration or deformities, and good flowering or yield characteristics.",
        "Cure": "No treatment required.",
        "Prevention": "Maintain proper spacing, nutrient balance, and pest/disease surveillance for early intervention."
    },
    "Potato___Early_Blight": {
        "Symptoms": "Dark, circular lesions with concentric rings (target-like) typically on older leaves. Yellowing may occur around lesions.",
        "Cure": "Apply protective fungicides such as Mancozeb or Chlorothalonil regularly, especially during warm, humid conditions.",
        "Prevention": "Practice crop rotation, avoid overhead irrigation, and remove plant debris after harvest."
    },
    "Potato___Healthy": {
        "Symptoms": "Vibrant green foliage, firm stems, and uniform tuber development without any signs of rot or discoloration.",
        "Cure": "No treatment necessary.",
        "Prevention": "Implement regular crop inspection and follow proper planting guidelines including spacing and nutrient application."
    },
    "Potato___Late_Blight": {
        "Symptoms": "Irregular brown lesions with pale green halos on leaves. Under moist conditions, white mold may appear on undersides of leaves.",
        "Cure": "Apply systemic fungicides such as Chlorothalonil or Metalaxyl. Remove and destroy infected plants immediately.",
        "Prevention": "Avoid wet foliage by using drip irrigation, ensure proper spacing, and use disease-resistant varieties."
    },
    "Red Rot": {
        "Symptoms": "Internal red discoloration in the stalk, foul odor, and external leaf drying. Stalk becomes hollow and collapses in advanced stages.",
        "Cure": "Uproot and burn affected canes. Apply fungicides recommended for sugarcane like Bavistin to reduce infection.",
        "Prevention": "Use disease-free setts, plant resistant varieties, and treat setts before planting using fungicide dips."
    },
    "Rice___Brown_Spot": {
        "Symptoms": "Brown, circular to oval spots with gray centers appear on leaves, grains, and seeds, causing seed discoloration and reduced quality.",
        "Cure": "Use fungicides like Tricyclazole or Propiconazole at the booting stage for effective control.",
        "Prevention": "Balance soil nutrients especially nitrogen and potassium, ensure seed treatment, and maintain proper plant spacing."
    },
    "Rice___Healthy": {
        "Symptoms": "Green upright leaves, firm stems, and even grain fill. No spots, discoloration, or stunted growth observed.",
        "Cure": "No cure required.",
        "Prevention": "Maintain regular irrigation schedule, apply fertilizers as per soil test, and scout fields regularly."
    },
    "Rice___Leaf_Blast": {
        "Symptoms": "Spindle-shaped lesions with gray centers and brown margins appear on leaves. Severe cases cause leaf death and reduced tillering.",
        "Cure": "Spray systemic fungicides like Carbendazim or Tricyclazole when early symptoms are visible.",
        "Prevention": "Use blast-resistant varieties, avoid excessive nitrogen application, and ensure good water drainage."
    },
    "Rice___Neck_Blast": {
        "Symptoms": "Panicle neck turns brown to black and shrivels, leading to incomplete grain filling and severe yield loss.",
        "Cure": "Apply systemic fungicides during panicle initiation stage such as Iprobenphos or Tricyclazole.",
        "Prevention": "Adopt balanced fertilizer usage, plant blast-resistant varieties, and ensure proper field sanitation."
    },
    "Wheat___Brown_Rust": {
        "Symptoms": "Rusty brown pustules develop in clusters primarily on leaf surfaces. In severe cases, leaves may dry out completely.",
        "Cure": "Timely foliar application of fungicides such as Propiconazole helps control the infection.",
        "Prevention": "Use resistant varieties, sow early to escape disease-prone periods, and maintain field sanitation."
    },
    "Wheat___Healthy": {
        "Symptoms": "Dark green, erect leaves with no spots, streaks, or signs of wilting. Uniform tillering and good ear development.",
        "Cure": "No action needed for healthy plants.",
        "Prevention": "Practice timely sowing, balanced nutrition, and monitor fields regularly for early detection."
    },
    "Wheat___Yellow_Rust": {
        "Symptoms": "Bright yellow-orange stripe-like pustules align parallel along the leaf veins. Infection can spread rapidly in cool, moist conditions.",
        "Cure": "Apply protective fungicides like Mancozeb or Tebuconazole as soon as symptoms appear.",
        "Prevention": "Cultivate resistant varieties, avoid late sowing, and apply nitrogen judiciously."
    }
}

# ============ Set Background ============
def set_bg(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Lexend:wght@400;600&display=swap');

        html, body, .stApp {{
            font-family: 'Lexend', sans-serif;
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .title-box {{
            background: rgba(255, 255, 255, 0.9);
            padding: 1.5rem;
            border-radius: 1rem;
            text-align: center;
            margin: 2rem 0;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}

        .result-box {{
            background-color: rgba(255, 255, 255, 0.85);
            border-radius: 1rem;
            padding: 2rem;
            text-align: left;
            box-shadow: 0 6px 15px rgba(0,0,0,0.1);
            margin-top: 2rem;
        }}

        .upload-btn {{
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 24px;
            border-radius: 8px;
            margin-top: 10px;
        }}

        h1, h2, h3 {{
            color: #2e3c5d;
        }}

        .metric {{
            font-size: 18px;
            margin-top: 10px;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ============ Page Config ============
st.set_page_config(page_title="🌿 Plant Disease Detection", page_icon="🌾", layout="centered")
set_bg("background.jpg")

# ============ Sidebar ============
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🧪 Detect", "ℹ️ About"])

# ============ Home ============
if page.startswith("🏠"):
    st.markdown("<div class='title-box'><h1>🌿 Plant Disease Detection</h1></div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Upload a leaf image to detect disease and get care tips instantly.</p>", unsafe_allow_html=True)
    st.image("hero_crop.jpg", use_column_width=True)

# ============ Detect ============
elif page.startswith("🧪"):
    st.markdown('<div class="title-box"><h2>📸 Upload a Leaf Image</h2></div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a crop leaf image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.write("🔄 Analyzing...")

        # Preprocess image
        img_resized = img.resize((128, 128))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_label = index_to_label[predicted_index]
        confidence_score = predictions[0][predicted_index] * 100

        info = disease_info.get(predicted_label, {})

        # Result Box
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown(f"<h3>🩺 Disease: <code>{predicted_label}</code></h3>", unsafe_allow_html=True)
        st.metric(label="Confidence Score", value=f"{confidence_score:.2f}%")
        st.markdown(f"**🧬 Symptoms:** {info.get('Symptoms', 'Not available')}")
        st.markdown(f"**💊 Cure:** {info.get('Cure', 'Not available')}")
        st.markdown(f"**🛡️ Prevention:** {info.get('Prevention', 'Not available')}")
        st.markdown('</div>', unsafe_allow_html=True)

# ============ About ============
elif page.startswith("ℹ️"):
    st.markdown('<div class="title-box"><h2>ℹ️ About This App</h2></div>', unsafe_allow_html=True)
    st.write("""
    This is an AI-powered crop disease detection app that helps farmers and agronomists detect plant diseases early using just leaf images.

    **Core Features:**
    - 🚀 Fast disease prediction using a CNN model
    - 🔍 Displays symptoms, cure, and prevention tips
    - 📷 Works with common leaf image formats
    - 🖼️ Stylish and responsive interface

    Built with **Streamlit**, **TensorFlow**, and ❤️.
    """)
