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

import src.utils as utils
from src.dataloader import DatasetSegmentation, collate_fn
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b
from src.lora import LoRA_sam

# 📌 Función para calcular IoU
def calculate_iou(pred, target):
    pred, target = pred.to(target.device), target
    pred = pred.bool()  # Convertir pred a booleano
    target = target.bool()  # Convertir target a booleano
    intersection = (pred & target).float().sum((1, 2))
    union = (pred | target).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

# 📌 Cargar configuración
with open("./config.yaml", "r") as ymlfile:
    config_file = yaml.load(ymlfile, Loader=yaml.Loader)

train_dataset_path = config_file["DATASET"]["TRAIN_PATH"]

metrics_path = "metrics.json"

# 📌 Cargar métricas previas si existen
try:
    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)
except FileNotFoundError:
    metrics_data = {"trains": []}

# 📌 Loop sobre diferentes valores de LoRA y batch size
for rank in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
    for batch_size in [4]:
        print(f"Rango:{rank} Batch:{batch_size}")

        sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
        sam_lora = LoRA_sam(sam, rank)
        model = sam_lora.sam
        processor = Samprocessor(model)
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
                iou = calculate_iou(preds, stk_gt)
                dice_score = (2 * (preds * stk_gt).sum()) / ((preds + stk_gt).sum() + 1e-6)
                accuracy = (preds == stk_gt).float().mean().item()

                epoch_ious.append(iou)
                epoch_dices.append(dice_score.item())
                epoch_accs.append(accuracy)

            # 📌 Guardar promedios de métricas
            avg_loss, avg_iou, avg_dice, avg_acc = mean(epoch_losses), mean(epoch_ious), mean(epoch_dices), mean(epoch_accs)
            total_loss.append(avg_loss)
            total_iou.append(avg_iou)
            total_dice.append(avg_dice)
            total_acc.append(avg_acc)

            print(f"EPOCH {epoch}: Loss={avg_loss:.4f}, IoU={avg_iou:.4f}, Dice={avg_dice:.4f}, Accuracy={avg_acc:.4f}")

        # 📌 Guardar métricas
        model_name = f"lora_rank{rank}_epochs{num_epochs}_batch{batch_size}.safetensors"
        metrics_data["trains"].append({
            "model": model_name,
            "loss": total_loss,
            "iou": total_iou,
            "dice": total_dice,
            "accuracy": total_acc
        })

        # 📌 Guardar modelo
        sam_lora.save_lora_parameters(model_name)
        # 🚨 Verificar que el modelo sigue bien después de guardar
        print(f"Modelo guardado: lora_rank{rank}_epochs{num_epochs}_batch{batch_size}.safetensors")
        print(f"Estado del modelo después de guardar: {sam_lora}")

        # Guardar métricas en JSON en cada iteración para evitar pérdidas
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=4)

# 📌 Guardar métricas en JSON
with open("metrics.json", "w") as f:
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
