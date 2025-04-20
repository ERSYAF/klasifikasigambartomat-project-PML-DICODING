# Tomato Leaf Disease Classification

Ini adalah proyek klasifikasi citra berbasis **Deep Learning** yang dikembangkan untuk submission pada kelas **Belajar Pengembangan Machine Learning** di Dicoding. Model ini dirancang untuk mengenali berbagai penyakit pada daun tomat berdasarkan gambar.

---

## Dataset: `plantvillage-dataset`

Dataset yang digunakan dalam proyek ini adalah **PlantVillage Dataset**, sebuah dataset populer untuk penelitian deteksi penyakit tanaman. Dari dataset lengkap yang mencakup **14 jenis tanaman dan 38 kelas penyakit**, proyek ini **hanya mengambil subset gambar tanaman tomat**. Gambar dipilih berdasarkan 4 kategori penyakit umum pada tomat.

### Distribusi Data

| Label                                | Jumlah Gambar |
|-------------------------------------|---------------|
| Tomato_Yellow_Leaf_Curl_Virus       | 5357          |
| Late_blight                          | 1909          |
| Spider_mites Two-spotted_spider_mite| 1676          |
| Septoria_leaf_spot                  | 1771          |
| **Total**                            | **10.713**    |

---

## Fitur dan Penyesuaian

Proyek ini telah disesuaikan agar memenuhi beberapa kriteria lanjutan, di antaranya:

- Menggunakan **Callback** untuk menghentikan pelatihan lebih awal (EarlyStopping) dan menyimpan model terbaik (ModelCheckpoint).
- Dataset terdiri dari lebih dari **10.000 gambar**.
- **Gambar tidak seragam ukurannya**.
- Jumlah label yang diklasifikasikan adalah **lebih dari 3 kelas**.
- Target akurasi untuk training dan validation adalah minimal **95%**.

---

## Arsitektur Model

Model menggunakan **MobileNetV2** sebagai feature extractor (pretrained dari ImageNet), lalu ditambahkan beberapa layer klasifikasi:

- `Conv2D(32)` + `ReLU` + `MaxPooling2D`
- `Conv2D(64)` + `ReLU` + `MaxPooling2D`
- `Flatten` + `Dropout(0.5)` + `Dense(128)`
- Output layer (`Dense`) dengan aktivasi `softmax` untuk klasifikasi 4 kelas

Model dibangun menggunakan `Keras Sequential` API, dan dilatih pada gambar berukuran **224x224 piksel**.

---

## Training Detail

Model dikompilasi menggunakan:

- Optimizer: `Adam`
- Loss function: `categorical_crossentropy`
- Metrics: `accuracy`

Training dilakukan selama maksimal 30 epoch dengan callback `EarlyStopping` yang menghentikan pelatihan jika akurasi validasi melebihi 97%.

---

## Inference

Contoh inference gambar tunggal:

```python
img_path = '/content/sample_data/test/Late_blight/sample.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

print("Prediksi:", class_labels[predicted_class])
```

---

## Evaluasi Prediksi

| Gambar | Label Asli                      | Prediksi Model                  |
|--------|----------------------------------|---------------------------------|
| 1      | Spider_mites Two-spotted_spider_mite | Spider_mites Two-spotted_spider_mite |
| 2      | Late_blight                     | Late_blight                     |
| 3      | Septoria_leaf_spot             | Septoria_leaf_spot              |
| 4      | Tomato_Yellow_Leaf_Curl_Virus  | Tomato_Yellow_Leaf_Curl_Virus   |

---

## Deployment dan Format Model

Model disimpan dalam tiga format:

- `SavedModel` (format standar TensorFlow)
- `TF Lite` untuk deployment mobile
- `TFJS` untuk deployment di web

---

