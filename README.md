# Laporan Proyek Machine Learning

### Nama : Ryan Anto Ramadhan

### NIM : 211351132

### Kelas : Pagi B

## Domain Proyek
Web app ini dikembangkan dengan maksud membantu ahli medis dalam melakukan diagnosa pasien diabetes. Web app ini bekerja menggunakan algorithma K-NearestNeighbors dengan menggunakan dataset dari kaggle sebagai datanya. Diproses dan dianalisis oleh saya sendiri.

## Business Understanding
Jumlah pasien yang mengidap diabetes semakin meningkat setiap tahun terutama pada negara-negara maju seperti Amerika Serikat dikarenakan pola makan dan pola hidup yang tidak teratur. Saking banyaknya pasien yang mengidap diabetes sehingga tidak semuanya berhasil diperiksa dan didiagnosa oleh ahlinya.

### Problem Statement
Banyak pasien yang tidak sempat terdiagnosa mengidap diabetes karena sang ahli kewalahan dengan jumlah pasien yang berdatangan.

### Goals
Lebih banyak pasien yang berhasil diperiksa dan diberi saran penanganan oleh sang ahli. 

### Solution Statements
- Membuat web app yang bisa memudahkan dan mempercepat proses pemeriksaan yang dilakukan oleh sang ahli pada pasiennya. 

## Data Understanding
Dataset ini berasal dari National Institute of Diabetes and Digestive and Kidney Diseases. Ianya tercipta dengan alasan untuk melakukan prediksi secara diagnostik apakah sang pasien memiliki penyakit diabetes atau tidak. Dataset ini memiliki 9 kolom dengan 768 baris data.
[Predict Diabetes](https://www.kaggle.com/datasets/whenamancodes/predict-diabities)

### Variabel-variabel pada Diabetes Prediction adalah sebagai berikut:
- Pregnancies : Menunjukkan jumlah kehamilan pasien. [int, 0 hingga 17]
- Glucose : Menunjukkan level glucose pada darah. [int, 0 hingga 199]
- BloodPressure : Menunjukkan tekanan darah pasien. [int, 0 hingga 122]
- SkinThickness : Menunjukkan ketebalan kulit pasien. [int, 0 hingga 99]
- Insulin : Menunjukkan level insulin pada darah pasien. [int, 0 hingga 846]
- BMI : Menunjukkan index massa badan pasien. [float, 0 hingga 67.1]
- DiabetesPedigreeFunction : Menunjukkan persentase diabetes [float, 0.08 hingga 2.42]
- Age : Menunjukkan umur pasien. [int, 21 hingga 81]
- Outcome : Menunjukkan hasil akhir. [1: Yes, 0: No]
## Data Preparation
### Import Dataset
Langkah pertama yang harus dilakukan adalah mengunduh datasetsnya. Berikut caranya.
```python
from google.colab import files
files.upload()
```
```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```
```python
!kaggle datasets download -d whenamancodes/predict-diabities
```
```python
!unzip predict-diabities.zip -d dataset
!ls dataset
```
### Import library yang diperlukan
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle
```

### Data Discovery
Memasukkan datasets kedalam dataframe yang bernama df menggunakan pandas.
```python
df = pd.read_csv("dataset/diabetes.csv")
```
Melihat 5 data pertama pada datasets.
```python
df.head()
```
Melihat informasi mengenai datasetsnya.
```python
df.info()
```
Terdapat 768 baris data dan mayoritas datatypenya adalah integer dan terdapat 2 datatype float.
```python
df.isnull().sum()
```
Bisa dilihat diatas, tidak terdapat nilai null pada datasetsnya.
```python
df.describe()
```
Melihat deskripsi datasets, mulai dari nilai min datasets hingga nilai max.
### Pre-Prosessing & EDA 
Langkah pertama yang akan saya lakukan pada tahap EDA ini adalah melihat korelasi antar kolomnya.
```python
correlation = df.corr()
sns.heatmap(correlation, annot = True)
```
![download](https://github.com/Ryan7445/diabetes-prediction/assets/149309065/7c7e921b-4e22-4e5f-8ce5-5fa40cb574c7)<br>
Terlihat korelasi yang cukup tinggi antara SkinThickness dan Insulin pada 44% serta korelasi Age dengan Pregnancies yaitu 54%.
```python
sns.countplot(x = 'Pregnancies', palette = 'Set2', data = df)
```
![download](https://github.com/Ryan7445/diabetes-prediction/assets/149309065/041e17bb-7428-4196-85b8-d734a0e2703c)<br>
Kita bisa melihat dari bar plot diatas terdapat beberapa orang yang pernah mengalami kehamilan sebanyak 17 kali, dan mayoritasnya pernah hamil satu kali.
```python
sns.countplot(x = 'Outcome', palette = 'Set2', data = df)
```
![download](https://github.com/Ryan7445/diabetes-prediction/assets/149309065/4ff2d15b-5448-49c7-88c5-42a6eb3d2c0b)<br>
Data Outcome(hasil) cenderung menunjukkan hasil 0 atau tidak memiliki diabetes. Hampir setengahnya merupakan orang yang memiliki diabetes.
```python
sns.boxplot(x=df["Glucose"])
```
![download](https://github.com/Ryan7445/diabetes-prediction/assets/149309065/eec880fc-0be6-4423-8a3e-6fb137d2166c)<br>
Kita bisa melihat pada kolom Glucose terdapat data outlier dengan nilai 0. Dan rata-rata nilai Glucose ini berada di kisaran 100 hingga 140an.
```python
sns.boxplot(x=df["BloodPressure"])
```
![download](https://github.com/Ryan7445/diabetes-prediction/assets/149309065/0f1a50c3-4823-49a6-8d4e-b843a66d7d56) <br>
Diatas merupakan box plot dari kolom BloodPressure, Rata-rata nilai BloodPressurenya adalah 60 hingga 80.
```python
sns.countplot(x = 'Pregnancies', hue = 'Outcome', palette = 'Set2', data = df)
```
![download](https://github.com/Ryan7445/diabetes-prediction/assets/149309065/dfd0f8bc-1a53-4b63-bf88-72e4fe8493d3)<br>
Kita bisa lihat diatas korelasi antara kolom Outcome dengan pregnancies. Dimana jumlah Pregnancies kemungkinan besar mempengaruhi hasil Outcome diabetes. <br> <br>
Kita akan lanjut ke tahap selanjutnya, yaitu Data Processing. <br> <br>
Langkah pertama saya adalah memasukkan fitur-fitur dan target pada variablenya.
```br
X = df.drop('Outcome', axis = 1)
y = df['Outcome']
```
Lalu saya akan menggunakan RandomOverSampler untuk mengisi jumlah baris data agar nilai outcomenya balance/seimbang antara 1 dan 0.
```python
ros = RandomOverSampler(random_state=0)
X, y = ros.fit_resample(X, y)
scaler = StandardScaler()
X_standard = scaler.fit_transform(X)
```
Kita juga menggunakan StandardScaler agar nilai-nilai pada setiap kolom memiliki nilai mean 0. <br>
Dan yang terakhir saya akan melakukan split data dengan nilai testnya 30% dari data dan sisanya dimasukkan pada training.
```python
X_train, X_test, y_train, y_test = train_test_split(X_standard, y, test_size = 0.3, random_state = 0)
```
## Modeling
Disini saya akan melakukan tahap modeling dengan menggunakan KNeighborsClassifier.
```python
knn = KNeighborsClassifier()
```
Saya akan mencari nilai n_neighbor yang terbaik untuk datasets ini.
```python
k_list = list(range(4,12))
k_values = dict(n_neighbors = k_list)
grid = GridSearchCV(knn, k_values, cv = 5, scoring = 'accuracy')
grid.fit(X_train, y_train)
GridSearchCV(cv=5, estimator=KNeighborsClassifier(),
             param_grid={'n_neighbors': [3, 4, 5, 6, 7, 8, 9, 11]},
             scoring='accuracy')
grid.best_params_, grid.best_score_
```
Bisa dilihat dari hasil diatas bahwa n_neighbors terbaik adalah 9 dengan score 73%. Mari kita gunakan nilai 9.
```python
knn = KNeighborsClassifier(n_neighbors = 9)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
```
Score yang kita dapatkan disini adalah 78%
### Visualisasi hasil algoritma
```
from sklearn import datasets, neighbors
from mlxtend.plotting import plot_decision_regions

def knn_comparison(data, k):
 x = data[['Pregnancies','BloodPressure']].values
 y = data['Outcome'].astype(int).values
 clf = neighbors.KNeighborsClassifier(n_neighbors=k)
 clf.fit(x, y)

# Plotting decision region
 plot_decision_regions(x, y, clf=clf, legend=2)

# Adding axes annotations
 plt.xlabel('Pregnancies')
 plt.ylabel('BloodPressure')
 plt.title('Knn dengan K='+ str(k))
 plt.show()

knn_comparison(df, 9)
```
![download](https://github.com/Ryan7445/diabetes-prediction/assets/149309065/fcd0fc2b-ab78-4731-9793-713ac81ec4dd)<br>
Diatas merupakan visualisasi knn antara Pregnancies dengan BloodPressure.
## Evaluation
Untuk tahap evaluasi saya menggunakan confusion matrix dan precision, recall dan f1-score untuk menguji modelnya. Ini sangat cocok untuk kasus-kasus yang membutuhkan validasi dalam pengklasifikasiannya. Kodenya sebagai berikut : 
```
print(classification_report(y_test, pred))
```
Tampaknya bagus, score yang didapatkan dari classification report adalah 78%, lebih tinggi dibanding menggunakan gridsearchcv.
```python
y_pred = knn.predict(X_test)
confusion_matrix(y_test, y_pred)
```
Diatas adalah hasil confusion matrixnya, dan menurut saya itu sudah cukup baik. Dengan 119 True Positive dan 116 True Negative serta 37 False Positive dan 28 False Negative
## Deployment
[Web App Prediksi Diabetes](https://diabetes-prediction-ryan.streamlit.app/) <br>
![image](https://github.com/Ryan7445/diabetes-prediction/assets/149309065/9c0f9faa-06fd-4333-bc17-fffef15d9d7c)

