import torch
from src.segment_anything import sam_model_registry, SamPredictor
from safetensors.torch import load_file
import matplotlib.pyplot as plt
import numpy as np

# 1. Carga del modelo base SAM
sam_checkpoint = "sam_vit_b_01ec64.pth"  # Reemplaza con la ruta correcta
model_type = "vit_b"  # Ajusta según el tipo de modelo SAM (vit_h, vit_l, vit_b)
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)

# 2. Carga de los pesos de LoRA
lora_checkpoint = "lora_rank512_epochs20_batch4_with_text.safetensors"  # Reemplaza con la ruta correcta

# 3. Combinación de modelos
# Asumiendo que tu modelo LoRA guarda los pesos en un diccionario 'lora_state_dict'
lora_state_dict = load_file(lora_checkpoint)

# Adaptar las claves del diccionario LoRA si es necesario
# Esto puede ser necesario si las claves no coinciden exactamente con las del modelo SAM base
# Ejemplo:
# from collections import OrderedDict
# new_lora_state_dict = OrderedDict()
# for k, v in lora_state_dict.items():
#     new_k = k.replace('module.', '') # Si hay un prefijo 'module.'
#     new_lora_state_dict[new_k] = v
# lora_state_dict = new_lora_state_dict

sam.load_state_dict(lora_state_dict, strict=False)  # Usa strict=False para permitir claves faltantes/adicionales

predictor = SamPredictor(sam)

# Preprocesamiento de la imagen
image = plt.imread("resize_IMG_4640.jpg.jpg")  # Reemplaza con la ruta correcta
predictor.set_image(image)

# Función para manejar los clics del usuario
def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x = int(event.xdata)
        y = int(event.ydata)

        # Crear prompt con el punto del clic
        input_point = np.array([[x, y]])
        input_label = np.array([1])  # 1 indica un punto positivo

        # Inferencia con SAM
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # Visualizar resultados
        plt.imshow(image)
        for mask in masks:
            plt.imshow(mask, alpha=0.5)
        plt.title(f"Clic en ({x}, {y})")
        plt.draw()

# Crear figura y conectar el evento de clic
fig, ax = plt.subplots()
ax.imshow(image)
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()