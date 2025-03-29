import torch
from safetensors.torch import load_file, save_file
from transformers import CLIPTokenizer, CLIPTextModel

# 📌 Archivo del modelo ya entrenado
model_path = "lora_rank512_epochs20_batch2.safetensors"

# 📌 Texto que queremos agregar como embedding
text_prompt = "camino de hormigas"

def generate_text_embedding(text):
    """ Genera el embedding de un texto usando CLIP """
    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    inputs = clip_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        text_embedding = clip_model(**inputs).last_hidden_state

    return text_embedding

# 📌 Generar embedding
text_embedding = generate_text_embedding(text_prompt)

# 📌 Cargar pesos del modelo desde el archivo .safetensors
model_weights = load_file(model_path)

# 📌 Agregar el embedding a los pesos del modelo
model_weights["text_embedding"] = text_embedding.squeeze(0)  # Guardamos sin la dimensión de batch

# 📌 Guardar el modelo modificado con el embedding incluido
new_model_path = "lora_rank512_epochs20_batch2_with_text.safetensors"
save_file(model_weights, new_model_path)

print(f"✅ Nuevo modelo guardado en {new_model_path} con embedding de '{text_prompt}'")
