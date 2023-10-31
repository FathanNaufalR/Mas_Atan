# Laporan Tugas Machine Learning
## Nama : Fathan Naufal R
## NIM : 211351054
## Kelas : Informatika Pagi B

## Domain Proyek
  Memprediksi harga Pizza dilhat dari, company,diameter, topping, variant, size, extra saus,dan extra keju nya. untuk itu saya berupaya membuat prediksi harga pizza karena banyaknya orang yang menyukai pizza.

## Business Understanding
 memudahkan orang orang,untuk mengetahui harga pizza dilhat dari, company,diameter, topping, variant, size, extra saus,dan extra keju nya.

### Problem Statements
- pelanggan harus datang langsung ke tempat penjualan pizza yang dimana akan memakan banyak biaya dan tenaga.

### Goals
- Membuat pelanggan bisa langsung melihat harga pizza yang bisa di akses dimanapun dan kapanpun

### Solution Statement
- Dapat memprediksi harga pizza dengan berbagai topping dan detail dari pizzanya (Dengan membandingkan dengan dataset)
- Membuat penelitian harga pizza dengan berbagai topping dan detail dari pizzanya (dengan algoritma R2)

## Data Understanding
https://www.kaggle.com/datasets/knightbearr/pizza-price-prediction/data

### Menentukan library yang di butuhkan
Pertama-tama kita akan mengimport library yang di butuhkan
``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
!kaggle datasets download -d knightbearr/pizza-price-prediction
```

Karena dataset yang terdownload berbentuk ZIP, maka kita Unzip terlebih dahulu datasetnya
```python
!mkdir pizza-price-prediction
!unzip pizza-price-prediction.zip -d pizza-price-prediction
!ls pizza-price-prediction
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

### Penjelasan Variabel pada pizza_price Dataset yaitu:
- company        : perusahaan yang menyediakan pizza (int64)
- price_rupiah   : harga dari setiap pizza yang di pesan (float64)
- diameter       : diameter pizza yang ingin di pesan (float64)
- topping        : topping yang akan dipilih pelanggan untuk pizza yang di pesan (int64)
- variant        : varian dari pizza yang akan di pesan contohnya, double_signature, american_favorite, super_supreme,DLL (int64)
- size           : ukuran dari pizza yang akan di pesan (int64)
- extra_sauce    : penawaran saus tambahan untuk pelanggan (int64)
- extra_cheese   : penawaran keju tambahan untuk pelanggan (int64)  


Mendrop kolom yang berisi NaN
```python
df.isna().sum()
```
```python
df=df.dropna()
```


## Menunjukan nilai unik pada kolom yang krusial dalam memprediksi harga
Yakni kolom  company,diameter, topping, variant, size, extra saus,dan extra keju
```python
df['company'].unique()
```
```python
df['price_rupiah'].unique()
```
```python
df['diameter'].unique()
```
```python
df['topping'].unique()
```
```python
df['variant'].unique()
```
```python
df['size'].unique()
```
```python
df['extra_sauce'].unique()
```
```python
df['extra_cheese'].unique()
```
Merubah Nilai kategorikal ke nilai numerikal pada kolom
```python
df['company'].replace(['A', 'B', 'C', 'D', 'E'],[0,1,2,3,4], inplace=True)
df['price_rupiah'].replace(['Rp235,000', 'Rp198,000', 'Rp120,000', 'Rp155,000', 'Rp248,000','Rp140,000', 'Rp110,000', 'Rp70,000', 'Rp90,000', 'Rp230,000','Rp188,000', 'Rp114,000', 'Rp149,000', 'Rp23,500', 'Rp46,000','Rp72,000', 'Rp49,000', 'Rp83,000', 'Rp96,000', 'Rp31,000','Rp69,000', 'Rp93,000', 'Rp75,000', 'Rp115,000', 'Rp123,000','Rp33,000', 'Rp76,000', 'Rp119,000', 'Rp126,500', 'Rp39,000','Rp99,000', 'Rp44,000', 'Rp78,000', 'Rp105,000', 'Rp35,000','Rp60,000', 'Rp98,000', 'Rp28,000', 'Rp51,000', 'Rp84,000','Rp32,000', 'Rp54,000', 'Rp92,000'],[235.000, 198.000, 120.000, 155.000, 248.000, 140.000, 110.000, 70.000, 90.000, 230.000, 188.000, 114.000, 149.000, 23.500, 46.000,72.000,49.000, 83.000, 96.000,31.000, 69.000, 93.000, 75.000, 115.000, 123.000, 33.000, 76.000, 119.000, 126.500, 39.000, 99.000,44.000, 78.000, 105.000, 35.000, 60.000, 98.000, 28.000, 51.000, 84.000, 32.000, 54.000, 92.000], inplace=True)
df['topping'].replace(['chicken', 'papperoni', 'mushrooms', 'smoked beef', 'mozzarella','black papper', 'tuna', 'meat', 'sausage', 'onion', 'vegetables','beef'],[0,1,2,3,4,5,6,7,8,9,10,11], inplace=True)
df['variant'].replace(['double_signature', 'american_favorite', 'super_supreme','meat_lovers', 'double_mix', 'classic', 'crunchy', 'new_york','double_decker', 'spicy_tuna', 'BBQ_meat_fiesta', 'BBQ_sausage','extravaganza', 'meat_eater', 'gournet_greek', 'italian_veggie','thai_veggie', 'american_classic', 'neptune_tuna', 'spicy tuna'],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], inplace=True)
df['size'].replace(['jumbo', 'reguler', 'small', 'medium', 'large', 'XL'],[1,2,3,4,5,6], inplace=True)
df['extra_sauce'].replace(['yes', 'no'],[0,1], inplace=True)
df['extra_cheese'].replace(['yes', 'no'],[0,1] , inplace=True)
```

Bentuk EDA
```python
sns.heatmap(df.isnull())
```
![image](https://github.com/FathanNaufalR/Mas_Atan/assets/149129682/d41a23e6-40b4-423e-b454-902637f46f3d)



```python
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)
```
![image](https://github.com/FathanNaufalR/Mas_Atan/assets/149129682/de3e86cb-022d-4b9f-af93-ea53cde35541)



```python
#Distribusi varian

plt.figure(figsize=(15,5))
sns.distplot(df['variant'])
```
![image](https://github.com/FathanNaufalR/Mas_Atan/assets/149129682/50d3c80c-568b-46fb-a693-bc0a0443cefa)


```python
#Distribusi topping

plt.figure(figsize=(15,5))
sns.distplot(df['topping'])
```
![image](https://github.com/FathanNaufalR/Mas_Atan/assets/149129682/d70fa363-9ee9-4fcc-82f4-94277a0c4a19)


## Modeling

Mengimpor train_test_split dari library sklearn dan Mengimpor LinearRegression dari library sklearn
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```
menyingkatkan LinearRegression menjadi variable lr
```python
lr = LinearRegression()
```

setelah sudah ditentukan kolom/atribut yang krusial , lalu drop kolom Price_rupiah (Yakni Variable dependen) pada dataframe
```python
X = df.drop(['price_rupiah'], axis=1)
```
Masukan kolom Price_rupiah pada variable y
```python
y = df['price_rupiah']
```

lakukan split data , untuk data train dan data test.
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

Membuat Regresi Linier
```python
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
score = lr.score(X_test,y_test)
print('Akurasi model regresi linier = ') ,score
```
Akurasi model regresi linier = 
(None, 0.786947701273773)

```python
input_data = np.array([[0,18.5,5,1,1,1,1]])
prediction = lr.predict(input_data)
print('Prediksi Harga Pizza: ', prediction)
```
Prediksi Harga PIZZA:  [163.04194142]

R-squared (R2) adalah ukuran statistik yang mewakili proporsi varians suatu variabel terikat yang dijelaskan oleh variabel bebas dalam model regresi.

```python
from sklearn.metrics import r2_score
r2_DT = r2_score(y_test, pred)  
r2_DT

print(f"Precision = {r2_DT}")
```
Precision = 0.786947701273773

## Saving Model
menyimpan file menjadi SAV
```python
import pickle
filename = 'pizza-price-prediction.sav'
pickle.dump(lr,open(filename,'wb'))
```

## Deployment

https://masatan-nfaps6gzqmkeahxtah3yf8.streamlit.app/

