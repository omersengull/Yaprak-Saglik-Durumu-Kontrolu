import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

# 1. Klasörlerden Veri Yükleme (Dengelenmiş veri seti için standart yükleyici)
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2 # %20'sini otomatik test için ayırır
)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

NUM_CLASSES = train_generator.num_classes

# 2. Standart Hazır Model Kurulumu
base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False # Model sabit kalacak, ezberleme yapamaz

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Eğitim 
print("Eğitim başlıyor...")
model.fit(
    train_generator, 
    epochs=3, 
    validation_data=validation_generator
)

# Modeli en temel formatta kaydediyoruz
model.save('plant_disease_model.keras')
print("✅ Eğitim bitti! 'plant_disease_model.keras' başarıyla oluşturuldu.")