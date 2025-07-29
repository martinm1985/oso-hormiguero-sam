import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import json
import yaml
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import CLIPTokenizer, CLIPTextModel

import src.utils as utils
from src.dataloader import DatasetSegmentation, collate_fn
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b
from src.lora import LoRA_sam

# 📌 Cargar modelo CLIP para generar embeddings de texto
clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def get_text_embeddings(text):
    inputs = clip_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = clip_model(**inputs).last_hidden_state
    return text_embeddings

# 📌 Modificar el procesador SAM para incluir embeddings de texto
class SamprocessorModified(Samprocessor):
    def __init__(self, model, text_prompt):
        super().__init__(model)
        self.text_prompt = text_prompt
        self.text_embedding = get_text_embeddings(self.text_prompt).to("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, image, original_size, box):
        processed_batch = super().__call__(image, original_size, box)  # Llamar a la versión original con los mismos argumentos
        processed_batch["text_embedding"] = self.text_embedding  # Agregar embedding al batch
        return processed_batch


# 📌 Cargar configuración
with open("./config.yaml", "r") as ymlfile:
    config_file = yaml.load(ymlfile, Loader=yaml.Loader)

train_dataset_path = config_file["DATASET"]["TRAIN_PATH"]
metrics_path = "metrics.json"

# 📌 Prompt de texto para el entrenamiento
text_prompt = "camino de hormigas"

# 📌 Cargar métricas previas si existen
try:
    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)
except FileNotFoundError:
    metrics_data = {"trains": []}

# 📌 Loop sobre diferentes valores de LoRA y batch size
for rank in [4, 8, 16, 32, 64, 128, 256, 512]:
    for batch_size in [4]:
        print(f"Rango:{rank} Batch:{batch_size}")

        sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
        sam_lora = LoRA_sam(sam, rank)
        model = sam_lora.sam
        processor = SamprocessorModified(model, text_prompt)
        train_ds = DatasetSegmentation(config_file, processor, mode="train")

        train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        optimizer = Adam(model.image_encoder.parameters(), lr=1e-4, weight_decay=0)
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        num_epochs = config_file["TRAIN"]["NUM_EPOCHS"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.train()
        model.to(device)

        total_loss, total_iou, total_dice, total_acc = [], [], [], []

        for epoch in range(num_epochs):
            epoch_losses, epoch_ious, epoch_dices, epoch_accs = [], [], [], []

            for batch in tqdm(train_dataloader):
                outputs = model(batched_input=batch, multimask_output=False)
                stk_gt, stk_out = utils.stacking_batch(batch, outputs)
                stk_out = stk_out.squeeze(1)
                stk_gt = stk_gt.unsqueeze(1)

                loss = seg_loss(stk_out, stk_gt.float().to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

                # 📌 Cálculo de métricas
                preds = (stk_out > 0.5).float().to(device)
                stk_gt = stk_gt.to(device)
                iou = (2 * (preds * stk_gt).sum()) / ((preds + stk_gt).sum() + 1e-6)
                accuracy = (preds == stk_gt).float().mean().item()

                epoch_ious.append(iou.item())
                epoch_dices.append(iou.item())  # Dice es lo mismo que IoU en este caso
                epoch_accs.append(accuracy)

            # 📌 Guardar métricas
            avg_loss, avg_iou, avg_dice, avg_acc = mean(epoch_losses), mean(epoch_ious), mean(epoch_dices), mean(epoch_accs)
            total_loss.append(avg_loss)
            total_iou.append(avg_iou)
            total_dice.append(avg_dice)
            total_acc.append(avg_acc)

            print(f"EPOCH {epoch}: Loss={avg_loss:.4f}, IoU={avg_iou:.4f}, Dice={avg_dice:.4f}, Accuracy={avg_acc:.4f}")

        # 📌 Guardar métricas y modelo
        model_name = f"lora_rank{rank}_epochs{num_epochs}_batch{batch_size}_with_text.safetensors"
        metrics_data["trains"].append({
            "model": model_name,
            "loss": total_loss,
            "iou": total_iou,
            "dice": total_dice,
            "accuracy": total_acc
        })

        sam_lora.save_lora_parameters(model_name)
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=4)

# 📌 Graficar métricas
plt.figure(figsize=(12, 5))
epochs_range = range(num_epochs)

plt.subplot(1, 2, 1)
plt.plot(epochs_range, total_loss, label="Loss")
plt.plot(epochs_range, total_dice, label="Dice Score")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, total_iou, label="IoU")
plt.plot(epochs_range, total_acc, label="Accuracy")
plt.legend()

plt.show()
