import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel
import yaml
import src.utils as utils  # Asumiendo que tienes utilidades para preprocesamiento, etc.
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b
from src.lora import LoRA_sam
from safetensors.torch import load_file

# 📌 Cargar configuración
with open("./config.yaml", "r") as ymlfile:
    config_file = yaml.load(ymlfile, Loader=yaml.Loader)

# 1. Cargar el modelo entrenado
rank = 2  # Ajustar según entrenamiento
model_name = "lora_rank512_epochs20_batch4_with_text.safetensors"
sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
sam_lora = LoRA_sam(sam, rank)
sam_lora.load_lora_parameters(model_name)
model = sam_lora.sam
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# 2. Cargar CLIP
clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
clip_model = clip_model.to(device)

clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def get_text_embeddings(text, device):
    """Obtiene embeddings de texto con CLIP"""
    inputs = clip_tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        text_embeddings = clip_model(**inputs).last_hidden_state
    return text_embeddings

# 3. Preprocesamiento de la imagen
image_path = "resize_IMG_4640.jpg.jpg"
image = Image.open(image_path).convert("RGB")
original_size = image.size

processor = Samprocessor(model)
input_image = processor(image, original_size=original_size, prompt=[])

# 🔹 Verificar `input_image`
if not isinstance(input_image, dict):
    raise TypeError(f"Error: Se esperaba un diccionario, pero se obtuvo {type(input_image)}.")

# 🔹 Convertir `original_size` de tupla a lista
if isinstance(input_image["original_size"], tuple):
    input_image["original_size"] = list(input_image["original_size"])

# 🔹 Eliminar `NoneType`
input_image = {k: v for k, v in input_image.items() if v is not None}

# 🔹 Mover tensores al dispositivo y agregar batch correctamente
for k, v in input_image.items():
    if isinstance(v, torch.Tensor):
        print(f"Clave: {k}, Shape antes: {v.shape}")  # 👀 Depuración
        if len(v.shape) == 3:  # Si es (C, H, W)
            input_image[k] = v.unsqueeze(0).to(device)  # Añadir batch dim -> (1, C, H, W)
        elif len(v.shape) == 4 and v.shape[0] == 1:  # Si ya tiene batch dim correcta
            input_image[k] = v.to(device)
        else:
            raise ValueError(f"Tensor con forma inesperada en clave {k}: {v.shape}")
    print(f"Clave: {k}, Shape después: {input_image[k].shape if isinstance(input_image[k], torch.Tensor) else type(input_image[k])}")  # 👀 Depuración

# 🔹 Asegurar que `input_image["image"]` tenga la forma correcta (1, 3, H, W)
if "image" in input_image and input_image["image"].shape[1] != 3:
    raise ValueError(f"Error: `image` debería tener 3 canales RGB, pero tiene {input_image['image'].shape[1]} canales.")

# 🔹 Convertir a lista para `model.forward`
batched_input = [input_image]

# 4. Generar el embedding de texto
text_prompt = "camino de hormigas"
text_embedding = get_text_embeddings(text_prompt, device)

# 5. Inferencia
with torch.no_grad():
    batched_input[0]["text_embedding"] = text_embedding.to(device)
    outputs = model(batched_input=batched_input, multimask_output=True)

# 🔹 Depuración: Ver qué tipo de salida devuelve el modelo
print(f"Tipo de outputs: {type(outputs)}")  # 👀
print(f"Contenido de outputs: {outputs}")  # 👀

# 🔹 Si outputs es una lista, tomar el primer elemento
if isinstance(outputs, list):
    outputs = outputs[0]

# 🔹 Verificar que ahora outputs es un diccionario
if not isinstance(outputs, dict):
    raise TypeError(f"Error: Se esperaba un diccionario, pero se obtuvo {type(outputs)} con contenido {outputs}")

# 🔹 Extraer máscaras correctamente
masks = outputs["masks"]

# 6. Postprocesamiento y visualización
masks = torch.nn.functional.interpolate(
    masks.float(),
    size=original_size,
    mode="bilinear",
    align_corners=False,
).cpu().numpy()
masks = masks > 0.5

plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks[0]:
    plt.imshow(mask, alpha=0.5)
plt.title(f"Segmentación con prompt: '{text_prompt}'")
plt.axis('off')
plt.show()
