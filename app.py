import streamlit as st
import cv2
import numpy as np
import pickle

# -------------------------------
# Load Trained Models
# -------------------------------
pca = pickle.load(open("pca.pkl", "rb"))
lda = pickle.load(open("lda.pkl", "rb"))
clf = pickle.load(open("ann_model.pkl", "rb"))
class_names = pickle.load(open("class_names.pkl", "rb"))

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎯 Face Recognition using PCA & ANN")
st.write("Upload a face image to identify the person.")

uploaded_file = st.file_uploader(
    "Upload a Face Image", 
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100, 100))
        flattened = resized.flatten().reshape(1, -1)

        # PCA → LDA → ANN
        pca_features = pca.transform(flattened)
        lda_features = lda.transform(pca_features)

        prediction = clf.predict(lda_features)
        probabilities = clf.predict_proba(lda_features)

        predicted_name = class_names[prediction[0]]
        confidence = np.max(probabilities) * 100

        st.image(img, caption="Uploaded Image", use_column_width=True)

        THRESHOLD = 65

        if confidence < THRESHOLD:
            st.warning("⚠ Unknown Person")
            st.info(f"Confidence: {confidence:.2f}%")
        else:
            st.success(f"Predicted Person: {predicted_name}")
            st.info(f"Confidence: {confidence:.2f}%")

    except Exception as e:
        st.error("Error occurred:")
        st.text(str(e))
