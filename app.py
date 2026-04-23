import streamlit as st
import torch
from PIL import Image
from model import imageModel

import base64

def set_background(image_file):
    with open(image_file, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{data}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    header {{visibility: hidden;}}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
set_background("websiteBackground.jpg")



st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.title("Deepfake Detector")
st.write("Upload an image to check if it's a deepfake.")
st.write("A project by Tariqul")


@st.cache_resource
def load_model():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model, transforms = imageModel(num_classes=2)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model, transforms, device


uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=400)

    if st.button("Run Deepfake Detection"):
        with st.spinner("Running deepfake detection now"):
            model, transforms, device = load_model()

            input_tensor = transforms(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs   = torch.softmax(outputs, dim=1)[0]
                pred_idx   = probs.argmax().item()
                confidence = probs[pred_idx].item()

        labels     = ["Deepfake", "Real"]
        prediction = labels[pred_idx]
        fake_prob  = probs[0].item()
        real_prob  = probs[1].item()

        st.success("Detection complete!")

        if prediction == "Deepfake":
            st.error(f" Prediction: this is a **{prediction}** image ")
        else:
            st.success(f" Prediction: this is a  **{prediction}** image")

        st.metric("Confidence", f"{confidence * 100:.1f}%")
        st.write("---")
        st.progress(fake_prob, text=f"Deepfake: {fake_prob*100:.1f}%")
        st.progress(real_prob, text=f"Real: {real_prob*100:.1f}%")