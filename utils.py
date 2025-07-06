import torch
import numpy as np
from tqdm import tqdm

from metrics import calculate_metrics_binary
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loss import ComboLossBinary
def train_one_epoch_binary(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_metrics = {
        'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 
        'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0, 'hausdorff': 0.0
    }
    num_batches = 0

    # Crear barra de progreso
    pbar = tqdm(dataloader, desc="Entrenando", leave=False)
    
    for batch in pbar:
        images = batch[0].to(device)
        masks = batch[1].to(device).float()

        if masks.dim() == 4 and masks.size(1) == 1:
            masks = masks.squeeze(1)

        outputs = model(images).squeeze(1)
        loss = loss_fn(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calcular métricas
        batch_metrics = calculate_metrics_binary(outputs, masks)
        
        total_loss += loss.item()
        for key in total_metrics:
            total_metrics[key] += batch_metrics[key]
        num_batches += 1

        # Actualizar barra de progreso
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Dice': f'{batch_metrics["dice"]:.4f}',
            'IoU': f'{batch_metrics["iou"]:.4f}'
        })

    # Calcular promedios
    avg_loss = total_loss / max(num_batches, 1)
    avg_metrics = {key: value / max(num_batches, 1) for key, value in total_metrics.items()}
    
    return avg_loss, avg_metrics


def evaluate_binary(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_metrics = {
        'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 
        'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0, 'hausdorff': 0.0
    }
    num_batches = 0

    # Crear barra de progreso
    pbar = tqdm(dataloader, desc="Evaluando", leave=False)

    with torch.no_grad():
        for batch in pbar:
            images = batch[0].to(device)
            masks = batch[1].to(device).float()

            if masks.dim() == 4 and masks.size(1) == 1:
                masks = masks.squeeze(1)

            outputs = model(images).squeeze(1)
            loss = loss_fn(outputs, masks)

            # Calcular métricas
            batch_metrics = calculate_metrics_binary(outputs, masks)
            
            total_loss += loss.item()
            for key in total_metrics:
                total_metrics[key] += batch_metrics[key]
            num_batches += 1

            # Actualizar barra de progreso
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{batch_metrics["dice"]:.4f}',
                'IoU': f'{batch_metrics["iou"]:.4f}'
            })

    # Calcular promedios
    avg_loss = total_loss / max(num_batches, 1)
    avg_metrics = {key: value / max(num_batches, 1) for key, value in total_metrics.items()}
    
    return avg_loss, avg_metrics

def train_model_binary(model, train_loader, test_loader, epochs: int = 10, lr: float = 1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loss_fn = ComboLossBinary(alpha=0.5, beta=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Almacenar historial de métricas
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': {'dice': [], 'iou': [], 'precision': [], 'recall': [], 'f1': [], 'accuracy': [], 'hausdorff': []},
        'val_metrics': {'dice': [], 'iou': [], 'precision': [], 'recall': [], 'f1': [], 'accuracy': [], 'hausdorff': []}
    }

    for epoch in range(epochs):
        print(f"\n📚 Época {epoch+1}/{epochs}")
        train_loss, train_metrics = train_one_epoch_binary(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_metrics = evaluate_binary(model, test_loader, loss_fn, device)

        # Scheduler: reducir LR si no mejora la validación
        scheduler.step(val_loss)

        # Guardar historial
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        for key in history['train_metrics']:
            history['train_metrics'][key].append(train_metrics[key])
            history['val_metrics'][key].append(val_metrics[key])

        print(f"🧠 Entrenamiento - Loss: {train_loss:.4f}, Dice: {train_metrics['dice']:.4f}, "
              f"IoU: {train_metrics['iou']:.4f}, F1: {train_metrics['f1']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"🧪 Validación   - Loss: {val_loss:.4f}, Dice: {val_metrics['dice']:.4f}, "
              f"IoU: {val_metrics['iou']:.4f}, F1: {val_metrics['f1']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

    return model, history