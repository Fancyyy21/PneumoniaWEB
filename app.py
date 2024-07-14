import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model = load_model('prediksi_pneumonia.h5')

# Define the path to save uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def load_and_preprocess_image(file):
    try:
        img = image.load_img(file, target_size=(150, 150))  # Adjust target size if necessary
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255.0  # Normalize the image data
        return x
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def get_description(label):
    if label == 'PNEUMONIA':
        return (
            "Pneumonia adalah infeksi pada paru-paru yang menyebabkan alveoli (kantung udara di paru-paru) "
            "terisi dengan cairan atau nanah, yang dapat mengganggu proses pertukaran oksigen dan karbon dioksida. "
            "Gejala pneumonia meliputi:\n"
            "- Batuk: Sering kali dengan dahak yang bisa berwarna kuning atau hijau.\n"
            "- Demam: Meningkatnya suhu tubuh yang disertai dengan menggigil.\n"
            "- Sesak Napas: Sulit bernapas atau napas yang cepat dan pendek.\n"
            "- Nyeri Dada: Nyeri yang terasa lebih buruk saat batuk atau bernapas dalam.\n"
            "- Kelelahan: Merasa lelah atau lemah tanpa alasan yang jelas.\n\n"
            "Pneumonia bisa disebabkan oleh berbagai mikroorganisme termasuk bakteri, virus, dan jamur. "
            "Diagnosis biasanya dilakukan melalui pemeriksaan fisik, rontgen dada, dan tes laboratorium. "
            "Perawatan tergantung pada penyebabnya dan mungkin melibatkan antibiotik, antivirus, atau antifungal."
        )
    elif label == 'NORMAL':
        return (
            "Paru-paru normal berarti berfungsi dengan baik dan tidak menunjukkan tanda-tanda infeksi atau "
            "kondisi medis lainnya yang mempengaruhi pernapasan. Ciri-ciri paru-paru yang sehat meliputi:\n"
            "- Pernapasan Lancar: Tidak ada kesulitan dalam bernapas, napas terasa normal dan tidak terengah-engah.\n"
            "- Tidak Ada Batuk Kronis: Tidak mengalami batuk terus-menerus atau batuk berdahak yang tidak hilang.\n"
            "- Tidak Ada Nyeri Dada: Tidak mengalami nyeri di daerah dada, terutama yang terasa lebih buruk saat bernapas atau batuk.\n"
            "- Energi dan Kesehatan Umum yang Baik: Merasa bugar dan energik tanpa gejala-gejala yang mencurigakan terkait dengan pernapasan.\n\n"
            "Paru-paru yang sehat mendukung aktivitas sehari-hari tanpa gangguan dan menjaga keseimbangan sistem pernapasan tubuh."
        )
    else:
        return "Deskripsi tidak tersedia untuk label yang diberikan."

def predict_image(file):
    try:
        x = load_and_preprocess_image(file)
        if x is not None:
            classes = model.predict(x)
            result = np.argmax(classes, axis=-1)[0]

            # Interpret the result and provide description
            if result == 0:
                prediction = "Normal"
                description = get_description('NORMAL')
            else:
                prediction = "Pneumonia"
                description = get_description('PNEUMONIA')
            
            return prediction, description, classes[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

# Streamlit app
st.title("Chest X-ray Pneumonia Detection")

# Sidebar
st.sidebar.title("Label Prediksi")
st.sidebar.write("Pilih gambar untuk diprediksi:")
st.sidebar.write("- 'Normal' untuk Paru-paru Normal")
st.sidebar.write("- 'Pneumonia' untuk Paru-paru yang Terkena Pneumonia")

file = st.file_uploader("Upload Gambar Yang Ingin Diprediksi", type=["jpg", "jpeg", "png"])

if file is not None:
    file_path = os.path.join(UPLOAD_FOLDER, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    st.image(file_path, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Hasil Prediksi :")
    prediction, description, probabilities = predict_image(file_path)
    if prediction:
        st.write(f"Prediction: {prediction}")
        st.write(description)

        # Tampilkan probabilitas prediksi di sidebar
    st.sidebar.write("### Probabilitas Prediksi")
    st.sidebar.write("- Normal:", probabilities[0])
    st.sidebar.write("- Pneumonia:", probabilities[1])

    st.sidebar.write(f"Prediksi: {prediction}")

