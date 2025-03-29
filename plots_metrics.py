import json
import matplotlib.pyplot as plt
import os

# Cargar el archivo JSON
with open("metrics.json", "r") as file:
    data = json.load(file)

# Crear carpeta "metrics" si no existe
output_dir = "metrics"
os.makedirs(output_dir, exist_ok=True)

# Extraer nombres de los modelos sin "lora_" y sin ".safetensors"
models = [
    train["model"].replace("lora_", "").removesuffix(".safetensors") 
    for train in data["trains"]
]

# Lista de métricas
metrics_names = ["loss", "iou", "dice", "accuracy"]

# Extraer valores finales (último epoch) de cada métrica para cada modelo
metrics_values = {metric: [train[metric][-1] for train in data["trains"]] for metric in metrics_names}

# Colores para diferenciar modelos
colors = ["red", "blue", "green", "purple", "orange", "cyan", "magenta"]

# Graficar cada métrica y guardar la imagen
for metric in metrics_names:
    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, metrics_values[metric], color=colors[:len(models)])

    # Agregar los valores al lado de los labels de los modelos con fuente más pequeña
    valores_redondeados = [f"{v:.3f}" for v in metrics_values[metric]]
    modelos_con_valores = [f"{m}\n({v})" for m, v in zip(models, valores_redondeados)]

    plt.xlabel("Modelos", fontsize=10)
    plt.ylabel("Valor", fontsize=10)
    plt.title(f"{metric.capitalize()} Final", fontsize=12)
    plt.ylim(0, 1.05)  # Ajustar el eje Y para evitar bordes

    # Aplicar una fuente más pequeña y espaciar bien los labels
    plt.xticks(range(len(models)), modelos_con_valores, rotation=45, ha="right", fontsize=7)

    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Guardar la gráfica en la carpeta "metrics"
    file_path = os.path.join(output_dir, f"{metric}.png")
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()  # Cerrar la figura para liberar memoria

print(f"Gráficas guardadas en la carpeta '{output_dir}'")
