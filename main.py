import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sklearn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from sklearn.model_selection import train_test_split
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import History
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV


# Definir los directorios de las imágenes
directorio_neumonia = "/Users/edssonestupinan/Downloads/chest_xray/rx/PNEUMONIA"
directorio_sin_neumonia = "/Users/edssonestupinan/Downloads/chest_xray/rx/NORMAL"

# Cargar las imágenes y etiquetas
imagenes_neumonia = [os.path.join(directorio_neumonia, img) for img in os.listdir(directorio_neumonia)]
imagenes_sin_neumonia = [os.path.join(directorio_sin_neumonia, img) for img in os.listdir(directorio_sin_neumonia)]

# Leer las imágenes y redimensionar a  128x128
X = []
y = []

for img_neumonia in imagenes_neumonia:
    img = cv2.imread(img_neumonia)
    img = cv2.resize(img, (128, 128))
    X.append(img)
    y.append(1)  # Etiqueta 1 para neumonía

for img_sin_neumonia in imagenes_sin_neumonia:
    img = cv2.imread(img_sin_neumonia)
    img = cv2.resize(img, (128, 128))
    X.append(img)
    y.append(0)  # Etiqueta 0 para sin neumonía

# Convierte las listas a arrays numpy
X = np.array(X)
y = np.array(y)


# Divide los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliza los datos
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255


#******* COnstrucción del MOdelo ***********


modelo = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compila el modelo
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrena el modelo
history=modelo.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

#*********** Evaluación del MOdelo ***********
# Evaluación del modelo en el conjunto de prueba
score = modelo.evaluate(X_test, y_test, verbose=0)
print("Pérdida en el conjunto de prueba:", score[0])
print("Precisión en el conjunto de prueba:", score[1])

# Predicciones en el conjunto de prueba
predicciones = modelo.predict(X_test)

# Genera gráficas
tf.keras.callbacks.History()
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(predicciones[y_test == 0], bins=30, label='Sin neumonía', alpha=0.7)
plt.hist(predicciones[y_test == 1], bins=30, label='Neumonía', alpha=0.7)
plt.xlabel('Confianza en la predicción')
plt.ylabel('Frecuencia')
plt.legend()
plt.show()

# Calcula la precisión del modelo
accuracy = accuracy_score(y_test, np.round(predicciones))
print("Precisión del modelo: {:.2f}%".format(accuracy * 100))

# Genera el informe de métricas de desempeño
informe_desempeno = classification_report(y_test, np.round(predicciones), target_names=['Sin Neumonía', 'Neumonía'])

# Imprime el informe de métricas de desempeño
print("Informe de Métricas de Desempeño:\n", informe_desempeno)
# Calcula la matriz de confusión
conf_matrix = confusion_matrix(y_test, np.round(predicciones))

# Visualiza la matriz de confusión como una gráfica
plt.figure(2,figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Sin Neumonía', 'Neumonía'], yticklabels=['Sin Neumonía', 'Neumonía'])
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión')
plt.show()

#*****************Guardar el modelo ************
# Guardar el modelo entrenado
#modelo.save('modelo_rayos_x.h5')

# Supongamos que tienes una imagen llamada 'imagen_rayos_x.jpg'
imagen_path = 'imagen_rayos_x.jpg'

# Cargar la imagen y preprocesarla
imagen = cv2.imread(imagen_path)
imagen = cv2.resize(imagen, (128, 128))  # Ajusta según las dimensiones del modelo
imagen = imagen.astype('float32') / 255

# Hacer la predicción con el modelo cargado
prediccion = modelo_cargado.predict(np.expand_dims(imagen, axis=0))

# Obtener la etiqueta de la predicción
etiqueta = "Neumonía" if prediccion > 0.5 else "Sin Neumonía"
print(f"La predicción es: {etiqueta}")
