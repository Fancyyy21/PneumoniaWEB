import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

import folium
from streamlit_folium import st_folium

# Load the trained model
model = load_model('prediksi_pneumonia.h5')

# Function to load and preprocess the image
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

# Function to get description based on the label
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

# Function to predict the image
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

# # Sidebar menu
# with st.sidebar:
#     selected = option_menu(
#         menu_title="Main Menu",  # Wajib
#         options=["Beranda", "Prediksi", "Kontak"],  # Wajib
#         icons=["house", "book", "envelope"],  # Wajib
#         menu_icon="cast",
#         default_index=0,
#     )

# Menu horizontal
selected = option_menu(
    menu_title="Menu Utama",  # Wajib
    options=["Beranda", "Prediksi", "Kontak", "About Us"],  # Wajib
    icons=["house", "book", "envelope", "person"],  # Opsional
    menu_icon="cast",  # Opsional
    default_index=0,  # Opsional
    orientation="horizontal"
)

if selected == "Beranda":
    st.title("Selamat Datang di Deteksi Pneumonia")
    st.write("""
    Aplikasi web ini menggunakan pembelajaran mesin untuk memprediksi pneumonia dari gambar X-ray dada.
    
    Fitur:
    - Unggah gambar X-ray dada
    - Dapatkan prediksi instan
    - Antarmuka yang mudah digunakan
    
    ### Informasi Tentang Paru-paru Normal dan Pneumonia
    
    **Paru-paru Normal:**
    Paru-paru yang sehat berfungsi untuk menyediakan oksigen ke darah dan membuang karbon dioksida. Gambar X-ray dada dari paru-paru yang sehat biasanya menunjukkan jaringan yang transparan dan homogen tanpa adanya bercak putih atau bayangan yang menunjukkan adanya cairan atau infeksi.

    **Paru-paru dengan Pneumonia:**
    Pneumonia adalah infeksi yang menyebabkan peradangan di kantung udara di salah satu atau kedua paru-paru. Kantung udara bisa terisi dengan cairan atau nanah, menyebabkan batuk dengan dahak atau nanah, demam, menggigil, dan kesulitan bernapas. Pada gambar X-ray, pneumonia dapat terlihat sebagai bercak putih atau bayangan yang menunjukkan adanya konsolidasi atau cairan di paru-paru. 

    **Gejala Pneumonia:**
    - Batuk yang dapat menghasilkan dahak
    - Demam, berkeringat, dan menggigil
    - Sesak napas
    - Nyeri dada yang semakin parah saat bernapas dalam atau batuk
    - Kelelahan dan kelelahan umum
    
    **Penyebab Pneumonia:**
    Pneumonia dapat disebabkan oleh berbagai organisme, termasuk bakteri, virus, dan jamur. Penyebab umum meliputi Streptococcus pneumoniae (bakteri), virus influenza, dan jamur seperti Pneumocystis jirovecii.

    **Pencegahan Pneumonia:**
    - Vaksinasi: Vaksin pneumokokus dan vaksin influenza dapat membantu mencegah beberapa jenis pneumonia.
    - Kebersihan: Mencuci tangan secara teratur dan menjaga kebersihan dapat mencegah penyebaran infeksi.
    - Gaya Hidup Sehat: Tidak merokok, menjaga kebugaran tubuh, dan menghindari paparan polusi udara dapat membantu menjaga kesehatan paru-paru.
    """)


if selected == "Prediksi":
    st.title("Deteksi Pneumonia")
    st.write("Unggah gambar X-ray dada untuk mendapatkan prediksi.")

    # File uploader
    file = st.file_uploader("Upload Gambar Yang Ingin Diprediksi", type=["jpg", "jpeg", "png"])

    if file is not None:
        file_path = os.path.join("uploads", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        st.image(file_path, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Hasil Prediksi :")
        prediction, description, probabilities = predict_image(file_path)
        
        if prediction:
            if prediction == "Normal":
                st.success(f"Prediction: {prediction}")
                st.sidebar.success(f"Prediksi: {prediction}")
            else:
                st.error(f"Prediction: {prediction}")
                st.sidebar.error(f"Prediksi: {prediction}")
            
            st.write(description)

            # Display prediction probabilities in the sidebar
            st.sidebar.write("### Probabilitas Prediksi")
            st.sidebar.write(f"- Normal: {probabilities[0]:.2f}")
            st.sidebar.write(f"- Pneumonia: {probabilities[1]:.2f}")



if selected == "Kontak":
    st.title("Hubungi Kami")
    st.write("Untuk pertanyaan lebih lanjut, silakan hubungi kami di:")
    st.write("""
    - Email: support@deteksi-pneumonia.com
    - Telepon: +62 123 456 789
    - Alamat: Jl. Sariasih No.54, Sarijadi, Kec. Sukasari, Kota Bandung, Jawa Barat 40151
    """)

    # Tentukan lokasi yang akan ditunjukkan pada peta
    location = [-6.874454127785651, 107.57568906598382]  # Koordinat untuk Bandung, Indonesia

    # Buat peta dengan Folium
    m = folium.Map(location=location, zoom_start=15)

    # Tambahkan marker pada peta
    folium.Marker(location, popup="Jl. Sariasih No.54, Sarijadi, Kec. Sukasari, Kota Bandung, Jawa Barat 40151").add_to(m)

    # Tampilkan peta dengan streamlit-folium
    st_folium(m, width=700, height=500)


if selected == "About Us":
    st.title("Tentang Kami")
    st.write("""
    ### Developer Aplikasi Deteksi Pneumonia

    **Nama:** Maulana Imanulhaq Nurdiana  
    **NPM:** 1214078
    """)
    
    st.image("aboutus/maul.jpeg", caption='Maulana Imanulhaq Nurdiana', width=150)

    st.write("""
    **Nama:** Raul Mahya Komaran  
    **NPM:** 1214054
    """)
    
    st.image("aboutus/raul.png", caption='Raul Mahya Komaran', width=150)

    st.write("""
    Kami adalah tim pengembang yang berdedikasi untuk menyediakan solusi kesehatan berbasis teknologi. Aplikasi deteksi pneumonia ini dirancang untuk membantu dalam diagnosis dini dan pengelolaan pneumonia, memanfaatkan teknologi pemrosesan gambar dan machine learning untuk mendeteksi tanda-tanda pneumonia pada gambar X-ray dada. Kami berharap aplikasi ini dapat memberikan kontribusi positif bagi masyarakat dalam menangani masalah kesehatan secara lebih efektif dan efisien.
    """)

