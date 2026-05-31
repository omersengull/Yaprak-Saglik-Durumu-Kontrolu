import os
import random

# Klasör yollarını belirliyoruz
HASTALIKLI_DIR = "dataset/train/Hastalikli"
SAGLIKLI_DIR = "dataset/train/Saglikli"

# Klasörlerdeki mevcut dosya listelerini alıyoruz
hastalikli_dosyalar = os.listdir(HASTALIKLI_DIR)
saglikli_dosyalar = os.listdir(SAGLIKLI_DIR)

hedef_sayi = len(saglikli_dosyalar) # Hedefimiz sağlıklı resim sayısı 
mevcut_hastalikli_sayisi = len(hastalikli_dosyalar)

print(f"Mevcut Sağlıklı Resim Sayısı: {hedef_sayi}")
print(f"Mevcut Hastalıklı Resim Sayısı: {mevcut_hastalikli_sayisi}")

if mevcut_hastalikli_sayisi > hedef_sayi:
    silinecek_adet = mevcut_hastalikli_sayisi - hedef_sayi
    print(f"Dengeyi sağlamak için {silinecek_adet} adet hastalıklı resim rastgele siliniyor...")
    
    # Resim listesini rastgele karıştırıyoruz (Sıralı silmemek için)
    random.shuffle(hastalikli_dosyalar)
    
    # Fazla olan resimleri seçiyoruz
    silinecek_resimler = hastalikli_dosyalar[:silinecek_adet]
    
    # Silme işlemi
    for resim in silinecek_resimler:
        dosya_yolu = os.path.join(HASTALIKLI_DIR, resim)
        if os.path.exists(dosya_yolu):
            os.remove(dosya_yolu)
            
    print(f"✅ İşlem tamamlandı! İki klasörde de artık tam olarak {hedef_sayi} adet resim var.")
else:
    print("Zaten hastalıklı resim sayısı sağlıklı resim sayısından fazla değil. Silme yapılmadı.")