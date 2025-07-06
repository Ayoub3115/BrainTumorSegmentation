import torch
from models import *
from preprocess import brain_dataloader
from utils import train_model_binary
from stats import print_final_metrics, plot_training_history, show_predictions_binary

# print("Comprobando GPU...")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Usando dispositivo: {device}")

# model = UNet(in_channels=3, out_channels=1).to(device)  # ⬅️ Salida binaria
# # Cargar los datasets
# img_dir = '/home/ahmedbegga/Escritorio/TFG /braintumorbueno/images'
# mask_dir = '/home/ahmedbegga/Escritorio/TFG /braintumorbueno/masks'

# train_loader, test_loader = brain_dataloader(img_dir, mask_dir, batch_size=32, num_workers=4)
# # ✅ CORRECCIÓN: Capturar el modelo entrenado y el historial
# trained_model, history = train_model_binary(model, train_loader, test_loader, epochs=45)

# # ✅ Ahora mostrar las métricas finales
# print_final_metrics(history)

# # ✅ Opcional: También puedes mostrar las gráficas del entrenamiento
# plot_training_history(history)

# # ✅ Opcional: Mostrar algunas predicciones
# show_predictions_binary(trained_model, test_loader, device, num_examples=3)

print("Comprobando GPU...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

model = UNetPlusPlus(in_channels=3, out_channels=1).to(device)  # ⬅️ Salida binaria

# Cargar los datasets
img_dir = '/home/ahmedbegga/Escritorio/TFG /braintumorbueno/images'
mask_dir = '/home/ahmedbegga/Escritorio/TFG /braintumorbueno/masks'

train_loader, test_loader = brain_dataloader(img_dir, mask_dir, batch_size=16, num_workers=8)
# ✅ CORRECCIÓN: Capturar el modelo entrenado y el historial
trained_model, history = train_model_binary(model, train_loader, test_loader, epochs=45)

# ✅ Ahora mostrar las métricas finales
print_final_metrics(history)

# ✅ Opcional: También puedes mostrar las gráficas del entrenamiento
plot_training_history(history)

# ✅ Opcional: Mostrar algunas predicciones
show_predictions_binary(trained_model, test_loader, device, num_examples=3)