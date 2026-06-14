import streamlit as st
import pickle

# Load the saved balanced model
@st.cache_resource
def load_model():
    with open('model_tokopedia_balanced.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Set up the UI
st.set_page_config(page_title="Sentimen Tokopedia", page_icon="🛒")
st.title("🚀 Analisis Sentimen Ulasan Tokopedia")
st.markdown("--- ")
st.write("Aplikasi ini menggunakan model **Naive Bayes** yang telah diseimbangkan untuk mendeteksi ulasan Positif, Negatif, atau Netral.")

# Text input for the review
ulasan_input = st.text_area("Masukkan teks ulasan pelanggan:", placeholder="Contoh: Barangnya jelek, pengiriman lama...")

# Prediction button
if st.button("Analisis Sekarang"):
    if ulasan_input.strip() == "":
        st.warning("Harap masukkan teks ulasan terlebih dahulu.")
    else:
        # Predict using the loaded model pipeline
        prediksi = model.predict([ulasan_input])[0]

        # Output visualization with styling
        st.subheader("Hasil Analisis:")
        if prediksi == 'Positif':
            st.success(f"🌟 Sentimen Prediksi: **{prediksi}**")
        elif prediksi == 'Negatif':
            st.error(f"⚠️ Sentimen Prediksi: **{prediksi}**")
        else:
            st.info(f"➖ Sentimen Prediksi: **{prediksi}**")
