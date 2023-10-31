# Laporan Tugas Machine Learning
## Nama : Fathan Naufal R
## NIM : 211351054
## Kelas : Informatika Pagi B

## Domain Proyek
  Memprediksi kanker payudara secara dini dapat membantu mengidentifikasi kasus kanker sebelum gejala muncul atau sebelum kanker mencapai tahap lanjut. Deteksi dini meningkatkan peluang kesembuhan dan mengurangi resiko penyebaran kanker pada organ lain, memprediksi mulai dari tekstur,area,kecekungan,radius, DLL. untuk mengidentifikasikan jenis kanker Ganas atau Jinak.

## Business Understanding
  Kanker merupakan penyakit yang sulit untuk di obati, karenanya perusahaan dan lembaga terlibat mengembangkan strategi bisnis yang lebih efektif, menciptakan produk dan layanan yang lebih baik, dan memberikan dukungan yang lebih baik bagi pasien dan individu yang terkena dampak kanker payudara. Juga mempermudah pembuatan obat jika sudah diketahui diagnosanya terlebih dahulu, maka dari itu dibuat lah prediksi jenis kanker payudara ini.

### Problem Statements
- Mahalnya biaya jika harus mengidetifikasikan jenis kanker, apakah kanker tersebut kanker ganas atau kanker jinak

### Goals
- Mencari cara untuk memprediksikan jenis kanker melalui diagnosa yang kita ketahui

### Solution Statement
- Dapat mengidentifikasikan jenis kanker yang dialami para pengidap dengan melihat diagnosanya (Dengan membandingkan dengan dataset)
- Membuat penelitian tentang kanker payudara untuk mengidentifikasikan jenisnya (dengan algoritma SVM)

## Data Understanding
(Breast Cancer Dataset [https://www.kaggle.com/datasets/imtkaggleteam/breast-cancer])

### Menentukan library yang di butuhkan
Pertama-tama kita akan mengimport library yang di butuhkan
``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
```
## Load Dataset
Selanjutnya saya mengupload file kaggle.json agar bisa mendapatkan akses pada kaggle
```python
from google.colab import files
files.upload()
```

Setelah itu saya membuat direktori dan izin akses pada skrip ini
```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```

Lalu mendownload Dataset yang sudah di pilih
```python
!kaggle datasets download -d imtkaggleteam/breast-cancer
```

Karena dataset yang terdownload berbentuk ZIP, maka kita Unzip terlebih dahulu datasetnya
```python
!mkdir breast-cancer
!unzip breast-cancer.zip -d breast-cancer
!ls breast-cancer
```

memanggil data CSV yang telah di unzip
```python
df = pd.read_csv('/content/breast-cancer/breast-cancer-wisconsin-data_data.csv')
```

Memunculkan data pada dataset dengan default 5 baris
```python
df.head()
```

Melihat ada berapa baris dan kolom
```python
df.shape
```

Mengetahui deskripsi pada data seperti tipedata
```python
df.describe()
```
```python
df.info()
```

Bentuk EDA
```python
sns.heatmap(df.isnull())
```
![Heatmap](https://github.com/FathanNaufalR/Mas_Atan/assets/149129682/1ef02b7c-a85c-4ed4-91e1-432092f3066e)


```python
plt.figure(figsize=(25,15))
sns.heatmap(df.corr(),annot=True)
```
![sns heatmap](https://github.com/FathanNaufalR/Mas_Atan/assets/149129682/8ab968c1-8f6d-4919-9b6c-b2af840d3a9d)


```python
models = df.groupby('diagnosis').count()[['smoothness_mean']].sort_values(by='smoothness_mean',ascending=True).reset_index()
models = models.rename(columns={'smoothness_mean':'rata-rata_kelembutan'})
```
```python
# B= benign(Jinak)
# M= Malignant(Ganas)

fig = plt.figure(figsize=(15,5))
sns.barplot(x=models['diagnosis'], y=models['rata-rata_kelembutan'], color='orange')
plt.xticks(rotation=60)
```
![plt diagnosis](https://github.com/FathanNaufalR/Mas_Atan/assets/149129682/fcaf744e-eefe-47c1-9b10-9d87a47db856)

```python
#Distribusi kecekungan terburuk
#concavity(Kecekungan)

plt.figure(figsize=(15,5))
sns.distplot(df['concavity_worst'])
```
![distribusi convacity_worst](https://github.com/FathanNaufalR/Mas_Atan/assets/149129682/5569e13c-e533-457f-ac51-8b27945249ce)


```python
# DISTRUBUSI Perimeter Terburuk

plt.figure(figsize=(15,5))
sns.distplot(df['perimeter_worst'])
```
![distribusi perimeter_worst](https://github.com/FathanNaufalR/Mas_Atan/assets/149129682/51fba6cd-8b60-42c6-b719-cc51d42b6aea)





### Penjelasan Variabel pada Breast Cancer Dataset yaitu:
-id                     : Nomor pengenal dari pasien  [Bertipe: int64]

-diagnosis              : Jenis dari kanker yang di idap kanker ganas atau jinak  [Bertipe: object]

-radius_mean            : Ini adalah rata-rata jarak dari pusat tumor ke tepi tumor  [Bertipe: float64]

-texture_mean           : Ini adalah rata-rata nilai intensitas tekstur sel-sel  [Bertipe: float64]

-perimeter_mean         : Ini adalah rata-rata panjang kontur tumor.  [Bertipe: float64]

-area_mean              : Ini adalah rata-rata area daerah dalam satu tumor. [Bertipe: float64]

-smoothness_mean        : Ini adalah rata-rata kehalusan permukaan sel-sel tumor. [Bertipe: float64]

-compactness_mean       : Ini adalah rata-rata seberapa kompak atau rapat sel-sel tumor. [Bertipe: float64]

-concavity_mean         : Ini adalah rata-rata sejauh mana tepi tumor cekung.  [Bertipe: float64]

-concave points_mean    : Ini adalah rata-rata jumlah titik cekung pada tepi tumor.  [Bertipe: float64]

-symmetry_mean          : Ini adalah rata-rata simetri sel-sel tumor. [Bertipe: float64]

-fractal_dimension_mean : Ini adalah rata-rata dimensi fraktal  [Bertipe: float64]

-radius_se              : Ini adalah kesalahan standar dari jarak dari pusat tumor ke tepi tumor.  [Bertipe: float64]

-texture_se             : Ini adalah kesalahan standar dari intensitas tekstur sel-sel dalam gambar.  [Bertipe: float64]

-perimeter_se           : Ini adalah kesalahan standar dari panjang kontur tumor.  [Bertipe: float64]

-area_se                : Ini adalah kesalahan standar dari area daerah dalam satu tumor. [Bertipe: float64]

-smoothness_se          : Ini adalah kesalahan standar dari kehalusan permukaan sel-sel tumor.  [Bertipe: float64]

-compactness_se         : Ini adalah kesalahan standar dari kompakitas sel-sel tumor.  [Bertipe: float64]

-concavity_se           : Ini adalah kesalahan standar dari sejauh mana tepi tumor cekung.  [Bertipe: float64]

-concave points_se      : Ini adalah kesalahan standar dari jumlah titik cekung pada tepi tumor.  [Bertipe: float64]

-symmetry_se            : Ini adalah kesalahan standar dari simetri sel-sel tumor.  [Bertipe: float64]

-fractal_dimension_se   : Ini adalah kesalahan standar dari dimensi fraktal sel-sel tumor.  [Bertipe: float64]

-radius_worst           : Ini adalah nilai terburuk (terbesar) dari jarak dari pusat tumor ke tepi tumor.  [Bertipe: float64]

-texture_worst          : Ini adalah nilai terburuk (terbesar) dari intensitas tekstur sel-sel dalam gambar.  [Bertipe: float64]

-perimeter_worst        : Ini adalah nilai terburuk (terbesar) dari panjang kontur tumor.  [Bertipe: float64]

-area_worst             : Ini adalah nilai terburuk (terbesar) dari area daerah dalam satu tumor.  [Bertipe: float64]

-smoothness_worst       : Ini adalah nilai terburuk (terbesar) dari kehalusan permukaan sel-sel tumor.  [Bertipe: float64]

-compactness_worst      : Ini adalah nilai terburuk (terbesar) dari kompakitas sel-sel tumor.  [Bertipe: float64]

-concavity_worst        : Ini adalah nilai terburuk (terbesar) dari sejauh mana tepi tumor cekung.  [Bertipe: float64]

-concave points_worst   : Ini adalah nilai terburuk (terbesar) dari jumlah titik cekung pada tepi tumor.  [Bertipe: float64]

-symmetry_worst         : Ini adalah nilai terburuk (terbesar) dari simetri sel-sel tumor.  [Bertipe: float64]

-fractal_dimension_worst: Ini adalah nilai terburuk (terbesar) dari dimensi fraktal sel-sel tumor. [Bertipe: float64]

-Unnamed: 32            : kolom yang tidak memiliki label atau informasi yang jelas dalam dataset. [Bertipe: float64]


Mendrop kolom yang berisi NaN agar tidak terjadi error pada saaat sedang memasukan kode classifier fit
```python
df = df.drop('Unnamed: 32', axis=1)
```

Masukan Fungsi ini di pakai untuk menghitung nilai yang ada di atribut "diagnosis", seperti menghitung jumlah pasien yang terkena kanker Benign(jinak) atau Malignant(Ganas) pada payudara
```python
df['diagnosis'].value_counts()
```

Pada fungsi ini saya akan memisahkan antara data lain(X) dan labelnya(diagnosis) menjadi (Y) nya
```python
#memisahkan data dan label
X = df.drop (columns='diagnosis', axis=1)
Y = df['diagnosis']
```
menampilkan Output dari data (X)
```python
print(X)
```

menampilkan Output label(Y), datanya sudah terpisah
```python
print(Y)
```
## Standarisasi data
Masukan fungsi scaler di gunakan untuk menstandarisasi data, dimana data yang akan saya standarisasi yaitu data (X)
```python
scaler = StandardScaler()
```

Masukan Fungsi untuk mentransformasi data (X)
```python
scaler.fit(X)
```
```python
standarized_data = scaler.transform(X)
```

menampilkan output dari fungsi standarisasi data (X),dimana angka angka pada data sudah tertransporm menjadi bentuk yang sudah terskala
```python
print(standarized_data)
```

pada kode ini kita mengecek apakah (X) sudah mejadi standarisasi atau belum
```python
X = standarized_data
Y = df['diagnosis']
```

menampilkan Output (X) menjadi standarisasi, dan label (Y)
```python
print(X)
print(Y)
```
## Memisahkan data train dan test
Masukan Fungsi untuk mengidentifikasi variabel training dan testing
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
```

Masukan Fungsi untuk memisahkan antara data training dan data testing, yaitu data trainingnya ada 455, dan data testingnya ada 114
```python
print(X.shape, X_train.shape, X_test.shape)
```

## Membuat data latih menggunakan SVM
Masukan Fungsi untuk mengimplementasikan atau memasukan algoritmah svm.SVC pada data X_train dan Y_train
```python
classifier = svm.SVC(kernel='linear')
```
```python
classifier.fit(X_train, Y_train)
```

## Membuat model evaluasi untuk mengukur tingkat akurasi
Masukan Fungsi untuk melihat akurasi training yang mencakup data X_train dan Y_train
```python
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
```

output pengecekan pada akurasi data train, yaitu sekitar 98%
```python
print('Akurasi data training adalah =',training_data_accuracy)
```
Akurasi data training adalah = 0.989010989010989

Masukan Fungsi untuk melihat akurasi pada data testing yaitu mencakup data X_test dan Y_test
```python
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
```

Output dari fungsi pengecekan akurasi data testing, akurasi yang di dapat yaitu sekitar 96%
```python
print('Akurasi data testing adalah =',test_data_accuracy)
```
Akurasi data testing adalah = 0.9649122807017544

## Membuat model prediksi
Kode ini di buat untuk mengecek hasil input pada dataset, dimana kita masukan data (X) dan lebel (Y) untuk melihat apakah data sesuai dengan datasetnya atau tidak
```python
input_data = (844981, 13, 21.82, 87.5, 519.8,	0.1273,	0.1932,	0.1859,	0.09353, 0.235,	0.07389, 0.3063, 1.002,	2.406, 24.32, 0.005731,	0.03502, 0.03553, 0.01226, 0.02143, 0.003749, 15.49, 30.73, 106.2, 739.3, 0.1703, 0.5401, 0.539, 0.206, 0.4378, 0.1072)

input_data_as_numpy_array = np.array(input_data)

input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

std_data = scaler.transform(input_data_reshape)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('kanker jinak')
else :
  print('kanker ganas')
```
[[-0.23638372 -0.32016686  0.58882978 -0.18408038 -0.38420727  2.20183876
   1.68400981  1.21909628  1.15069158  1.96559991  1.57246173 -0.35685002
  -0.389818   -0.2277434  -0.35240312 -0.43667734  0.53329023  0.12056834
   0.07524305  0.10748154 -0.0173632  -0.1613566   0.82281333 -0.03160911
  -0.24836341  1.66275699  1.81830968  1.28003453  1.39161624  2.38985717
   1.28864955]]
   
['M']

kanker ganas


jika pada saat pengecekan terjadi error pada label dikarenakan lebel pada data saya bertipe data String, bisa di ubah menjadi int64 dengan menggunakan kode berikut.
```python
oe=OrdinalEncoder(categories=[['B','M']])
col=['diagnosis']
for col_n in col:
  df[col_n]=oe.fit_transform(df[[col_n]])
```

## Simpan Model
Masukan Fungsi untuk menyimpan file prediksi jenis kanker payudara berjenis SAV,untuk nanti saya pakai pada streamlit
```python
import pickle
```
```python
filename = 'Prediksi_Jenis_kanker_payudara.sav'
pickle.dump(classifier, open(filename,'wb'))
```

## Deployment

https://masatan-prediksi-jenis-kanker-payudara.streamlit.app/

![Streamlit](https://github.com/FathanNaufalR/Mas_Atan/assets/149129682/eed910a4-76c2-48d6-9e86-fa88cdfba3fa)

