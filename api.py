from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import io
from tensorflow.keras.models import load_model
import cv2

app = FastAPI()

# Cargar el modelo previamente entrenado (reemplaza 'modelo_rayos_x.h5' con tu modelo real)
modelo = load_model('modelo_rayos_x.h5')

def preprocesar_imagen(imagen):
    # Ajusta la forma y la ordenación de las dimensiones
    imagen = cv2.resize(imagen, (128, 128))  # Ajusta según las dimensiones del modelo
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)  # Convierte a formato RGB
    imagen = imagen.astype('float32') / 255
    return imagen

@app.post("/predecir/")
async def predecir_neumonia(imagen: UploadFile):
    contenido = await imagen.read()
    imagen_pil = Image.open(io.BytesIO(contenido))
    imagen_array = np.array(imagen_pil)
    imagen_preprocesada = preprocesar_imagen(imagen_array)  # Preprocesar la imagen

    # Realizar predicción usando el modelo cargado
    resultado = modelo.predict(np.expand_dims(imagen_preprocesada, axis=0))

    # Obtener la etiqueta de la predicción
    etiqueta = "Neumonía" if resultado[0, 0] > 0.5 else "Sin Neumonía"
    mensaje = f"La predicción es: {etiqueta}"
    print(mensaje)

    # Devolver la predicción como respuesta
    return {"es_neumonia": etiqueta == "Neumonía", "mensaje": mensaje}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
