import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="AI Scene Classifier", layout="wide")

# ---------- UI STYLE ----------
st.markdown("""
<style>

body {
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}

.title{
font-size:38px;
font-weight:700;
text-align:center;
color:white;
margin-bottom:25px;
}

.card{
background: rgba(255,255,255,0.08);
border-radius:15px;
padding:20px;
border:1px solid rgba(255,255,255,0.2);
box-shadow:0 8px 30px rgba(0,0,0,0.35);
}

.section{
background: rgba(255,255,255,0.08);
border-radius:15px;
padding:20px;
margin-top:20px;
border:1px solid rgba(255,255,255,0.2);
}

.top-item{
font-size:18px;
margin-top:8px;
}

</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
model = tf.keras.models.load_model("model.h5")

classes = ['buildings','forest','glacier','mountain','sea','street']

# ---------- COLOR MAP ----------
color_map = {
    "buildings": "linear-gradient(135deg,#6c757d,#adb5bd)",
    "forest": "linear-gradient(135deg,#134e5e,#71b280)",
    "glacier": "linear-gradient(135deg,#83a4d4,#b6fbff)",
    "mountain": "linear-gradient(135deg,#834d1e,#d04a02)",
    "sea": "linear-gradient(135deg,#1e3c72,#2a5298)",
    "street": "linear-gradient(135deg,#232526,#414345)"
}

st.markdown('<div class="title">AI Scene Image Classifier</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

# ---------- PREPROCESS ----------
def preprocess(img):

    img = img.convert("RGB")
    img = img.resize((224,224))

    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    return img


if uploaded_file is not None:

    img = Image.open(uploaded_file)

    img_array = preprocess(img)

    with st.spinner("Analyzing image..."):
        prediction = model.predict(img_array)[0]

    predicted_class = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction))*100

    card_color = color_map[predicted_class]

    # ---------- HORIZONTAL LAYOUT ----------
    col1, col2 = st.columns([1,1])

    # IMAGE PANEL
    with col1:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.image(img, width="stretch")

        st.markdown('</div>', unsafe_allow_html=True)

    # RESULT PANEL
    with col2:

        st.markdown(f"""
        <div style="
        background:{card_color};
        border-radius:15px;
        padding:30px;
        text-align:center;
        color:white;
        font-weight:700;
        box-shadow:0 10px 30px rgba(0,0,0,0.35);
        ">

        <div style="font-size:18px;">
        Prediction Result
        </div>

        <div style="font-size:42px; letter-spacing:2px;">
        {predicted_class.upper()}
        </div>

        <div style="font-size:18px; margin-top:10px;">
        Confidence Score: {confidence:.2f}%
        </div>

        </div>
        """, unsafe_allow_html=True)

        # ---------- CONFIDENCE GAUGE ----------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={'text': "Confidence"},
            gauge={
                'axis': {'range':[0,100]},
                'bar': {'color':"#00ffd5"}
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

    # ---------- PROBABILITY BAR CHART ----------
    st.markdown('<div class="section">', unsafe_allow_html=True)

    st.subheader("Class Probabilities")

    prob_df = pd.DataFrame({
        "Scene":classes,
        "Probability":prediction
    })

    st.bar_chart(prob_df.set_index("Scene"), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- TOP PREDICTIONS ----------
    st.markdown('<div class="section">', unsafe_allow_html=True)

    st.subheader("Top Predictions")

    top3 = prob_df.sort_values(
        "Probability",
        ascending=False
    ).head(3)

    for i,row in top3.iterrows():
        st.markdown(
            f'<div class="top-item">{row["Scene"]} — {row["Probability"]*100:.2f}%</div>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- DOWNLOAD RESULTS ----------
    st.markdown('<div class="section">', unsafe_allow_html=True)

    result_df = pd.DataFrame({
        "Scene":classes,
        "Probability (%)":prediction*100
    })

    csv = result_df.to_csv(index=False)

    st.download_button(
        label="Download Prediction Results",
        data=csv,
        file_name="prediction_results.csv",
        mime="text/csv"
    )

    st.markdown('</div>', unsafe_allow_html=True)