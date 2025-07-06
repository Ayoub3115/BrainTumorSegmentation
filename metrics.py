import torch
import numpy as np
from typing import Dict
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import precision_score, recall_score, f1_score
import cv2

def dice_score_binary(preds: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> float:
    preds = torch.sigmoid(preds)
    preds = (preds >= 0.68).float()
    targets = targets.float()

    intersection = (preds * targets).sum().item()
    union = preds.sum().item() + targets.sum().item()

    if union == 0:
        return 1.0  # Si no hay nada en la máscara y predicción, es perfecto
    return (2. * intersection + epsilon) / (union + epsilon)


def iou_score_binary(preds: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    Calcula el Intersection over Union (IoU) para segmentación binaria
    """
    preds = torch.sigmoid(preds)
    preds = (preds >= 0.68).float()
    targets = targets.float()

    intersection = (preds * targets).sum().item()
    union = preds.sum().item() + targets.sum().item() - intersection

    if union == 0:
        return 1.0  # Si no hay nada en la máscara y predicción, es perfecto
    return (intersection + epsilon) / (union + epsilon)


def hausdorff_distance_binary(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calcula la distancia de Hausdorff para segmentación binaria
    """
    try:
        # Convertir a arrays numpy
        preds_np = torch.sigmoid(preds).cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # Convertir a binario
        preds_np = (preds_np >= 0.68).astype(np.uint8)
        targets_np = targets_np.astype(np.uint8)
        
        # Si es un batch, procesar solo la primera imagen
        if preds_np.ndim > 2:
            preds_np = preds_np[0] if preds_np.ndim == 3 else preds_np[0, 0]
            targets_np = targets_np[0] if targets_np.ndim == 3 else targets_np[0, 0]
        
        # Encontrar contornos
        pred_contours, _ = cv2.findContours(preds_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        target_contours, _ = cv2.findContours(targets_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Si no hay contornos en alguna de las imágenes
        if len(pred_contours) == 0 or len(target_contours) == 0:
            if len(pred_contours) == 0 and len(target_contours) == 0:
                return 0.0  # Ambas vacías, distancia perfecta
            else:
                return float('inf')  # Una vacía, otra no - distancia infinita
        
        # Obtener puntos de los contornos
        pred_points = np.concatenate([contour.reshape(-1, 2) for contour in pred_contours])
        target_points = np.concatenate([contour.reshape(-1, 2) for contour in target_contours])
        
        # Calcular distancia de Hausdorff
        hd1 = directed_hausdorff(pred_points, target_points)[0]
        hd2 = directed_hausdorff(target_points, pred_points)[0]
        hd = max(hd1, hd2)
        
        return hd
        
    except Exception as e:
        # En caso de error, retornar un valor alto pero finito
        return 100.0


def calculate_metrics_binary(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Calcula múltiples métricas para segmentación binaria incluyendo IoU y HD
    """
    with torch.no_grad():
        # Convertir a probabilidades y predicciones binarias
        probs = torch.sigmoid(preds).cpu().numpy().flatten()
        preds_binary = (probs > 0.68).astype(int)
        targets_flat = targets.cpu().numpy().flatten().astype(int)
        
        # Evitar división por cero
        epsilon = 1e-7
        
        # Dice Score
        intersection = np.sum(preds_binary * targets_flat)
        union = np.sum(preds_binary) + np.sum(targets_flat)
        dice = (2. * intersection + epsilon) / (union + epsilon) if union > 0 else 1.0
        
        # IoU Score
        union_iou = np.sum(preds_binary) + np.sum(targets_flat) - intersection
        iou = (intersection + epsilon) / (union_iou + epsilon) if union_iou > 0 else 1.0
        
        # Precision, Recall, F1
        try:
            precision = precision_score(targets_flat, preds_binary, zero_division=0)
            recall = recall_score(targets_flat, preds_binary, zero_division=0)
            f1 = f1_score(targets_flat, preds_binary, zero_division=0)
        except:
            precision = recall = f1 = 0.0
        
        # Accuracy
        try:
            correct_predictions = np.sum(preds_binary == targets_flat)
            total_predictions = len(targets_flat)
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        except:
            accuracy = 0.0
        
        # Hausdorff Distance
        try:
            hd = hausdorff_distance_binary(preds, targets)
            # Normalizar HD para que valores más bajos sean mejores
            # y estén en un rango razonable
            hd = min(hd, 100.0)  # Cap a 100 para evitar valores extremos
        except:
            hd = 100.0
        
        return {
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'hausdorff': hd
        }