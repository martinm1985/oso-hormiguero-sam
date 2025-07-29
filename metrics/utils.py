import os
import torch
from PIL import Image
import numpy as np
from src.segment_anything.predictor import SamPredictor
from torchvision import transforms
import matplotlib.pyplot as plt
import os  # Importa la librería os para trabajar con rutas y carpetas


def getImagesAndGroundTruthMasks(ruta_imagenes, ruta_mascaras):

    imagenes = []
    mascaras_ground_truth = []

    for nombre_archivo in os.listdir(ruta_imagenes):
        if nombre_archivo.endswith((".jpg", ".png")):  # Ajusta las extensiones
            ruta_imagen = os.path.join(ruta_imagenes, nombre_archivo)
            imagen = Image.open(ruta_imagen).convert("RGB")  # Convierte a RGB si es necesario
            print("Forma de la imagen original:", np.array(imagen).shape)
            imagenes.append(imagen)

    for nombre_archivo in os.listdir(ruta_mascaras):
        if nombre_archivo.endswith((".jpg", ".png")):  # Ajusta las extensiones
            ruta_mascara = os.path.join(ruta_mascaras, nombre_archivo)
            mascara = Image.open(ruta_mascara).convert("L")  # Convierte a escala de grises para la máscara
            print("Forma de la máscara ground truth (antes de convertir a numpy):", np.array(mascara).shape)
            mascaras_ground_truth.append(np.array(mascara)) # Convierte a numpy array

    return imagenes, mascaras_ground_truth


def getPredictMasks(sam, imagenes, dispositivo):
    predictor = SamPredictor(sam)

    mascaras_predichas = []

    # ***CREACIÓN DE LA CARPETA***
    ruta_carpeta = "./predicciones"  # Reemplaza con la ruta deseada
    os.makedirs(ruta_carpeta, exist_ok=True)  # Crea la carpeta si no existe

    transform = transforms.ToTensor() #Transformacion a tensor

    for i, imagen in enumerate(imagenes):  # imagenes es una lista de PIL Images
        imagen_tensor = transform(imagen).to(dispositivo)  # Convierte a tensor y mueve a dispositivo

        predictor.set_image(imagen_tensor)  # Set image espera tensores de 4 dimensiones (N, C, H, W)

        masks, _, _ = predictor.predict()

        # ***CONVERSIÓN A TENSOR ANTES DE .cpu()***
        masks_tensor = torch.from_numpy(masks).to(dispositivo)  # Convierte a tensor y mueve a dispositivo

        # ***BINARIZACIÓN DE LA MÁSCARA PREDECIDA***
        umbral = 0.5  # Ajusta este valor si es necesario
        mascara_binaria = (masks_tensor > umbral).cpu().numpy().astype(np.uint8)  # Binariza y convierte a NumPy

        mascaras_predichas.append(mascara_binaria)
    
     # ***GUARDADO DE LA MÁSCARA PREDECIDA***
    for j, mask in enumerate(mascaras_predichas):  # Itera sobre las máscaras predichas
        nombre_archivo = f"mascara_{i+1}_{j+1}.png"  # Nombre del archivo (puedes personalizarlo)
        ruta_archivo = os.path.join(ruta_carpeta, nombre_archivo)  # Ruta completa al archivo

        plt.imsave(ruta_archivo, mask, cmap='gray')  # Guarda la imagen (sin mostrarla)
    
    return mascaras_predichas