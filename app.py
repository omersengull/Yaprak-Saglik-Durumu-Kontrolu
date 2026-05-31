import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.title("🌱 Yaprak Sağlık Durumu Kontrol Sistemi")
st.write("Yüklediğiniz yaprak fotoğrafının üzerindeki hastalık belirtilerini analiz eder.")

@st.cache_resource
def model_yukle():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'plant_disease_model.keras')
    model = tf.keras.models.load_model(model_path)
    # Klasörlerin alfabetik sırası: Hastalikli (0), Saglikli (1)
    classes = ['Hastalıklı Yaprak', 'Sağlıklı Yaprak'] 
    return model, classes

try:
    model, class_names = model_yukle()
except Exception as e:
    st.error(f"Model yüklenemedi! Hata: {e}")
    st.stop()

uploaded_file = st.file_uploader("Analiz için yaprak fotoğrafı seçiniz...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Analiz Edilen Yaprak', use_column_width=True)
    st.write("Yaprak yüzeyi taranıyor...")
    
    img = image.resize((224, 224))
    img_array = np.array(img)
    
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    score = predictions[0]
    
    tahmin_indeksi = np.argmax(score)
    tahmin_edilen_durum = class_names[tahmin_indeksi]
    dogruluk_orani = 100 * np.max(score)
    
    st.write("---")
    st.subheader("📊 Analiz Sonucu:")
    
    # Sağlık durumuna göre dinamik arayüz renklendirmesi
    if tahmin_indeksi == 0: # Hastalıklı
        st.error(f"🚨 **Teşhis:** {tahmin_edilen_durum}  \n**Tespit Edilme Güven Oranı:** %{dogruluk_orani:.2f}")
        st.warning("⚠️ **Tavsiye:** Yaprak üzerinde leke veya çürüme saptandı. Diğer bitkilere yayılmaması için ilaçlama yapılması veya enfekte bölgenin uzaklaştırılması önerilir.")
    else: # Sağlıklı
        st.success(f"✅ **Teşhis:** {tahmin_edilen_durum}  \n**Güven Oranı:** %{dogruluk_orani:.2f}")
        st.info("💡 **Durum:** Bitki yaprağında herhangi bir patojen veya hastalık belirtisine rastlanmadı. Mevcut sulama ve bakım rutinine devam edebilirsiniz.")