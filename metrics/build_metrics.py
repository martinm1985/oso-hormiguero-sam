from sklearn.metrics import jaccard_score  # Ejemplo: Índice de Jaccard (IoU)
import numpy as np

def calcular_iou(mascara_predicha, mascara_ground_truth):
    # Asegúrate de que las máscaras sean booleanas (True/False)
    print("Forma de mascara_ground_truth:", mascara_ground_truth.shape)
    print("Forma de mascara_predicha:", mascara_predicha.shape)
    mascara_predicha = mascara_predicha > 0  # Ajusta el umbral si es necesario
    mascara_ground_truth = mascara_ground_truth > 0

    ious = []
    for mascara_sam in mascara_predicha:
        # (código para binarizar y convertir a booleano, como en el ejemplo anterior)
        iou = jaccard_score(mascara_ground_truth.flatten(), mascara_sam.flatten())
        ious.append(iou)
    return max(ious)

def calculo_de_iou(mascaras_predichas, mascaras_ground_truth):
    ious = []
    for predicha, gt in zip(mascaras_predichas, mascaras_ground_truth):
        iou = calcular_iou(predicha, gt)
        ious.append(iou)

    iou_promedio = np.mean(ious)
    print(f"IoU promedio: {iou_promedio}")


# Otras métricas que puedes considerar:
# - Dice coefficient
# - Precision y Recall
# - F1-score
# - Hausdorff distance