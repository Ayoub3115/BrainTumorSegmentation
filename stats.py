import matplotlib.pyplot as plt
from metrics import calculate_metrics_binary
import torch
def print_final_metrics(history):
    """
    Imprime un resumen final de las métricas
    """
    print("\n" + "="*80)
    print("📊 RESUMEN FINAL DE MÉTRICAS")
    print("="*80)
    
    final_train = {key: values[-1] for key, values in history['train_metrics'].items()}
    final_val = {key: values[-1] for key, values in history['val_metrics'].items()}
    
    print(f"🚂 ENTRENAMIENTO:")
    print(f"   Dice Score:  {final_train['dice']:.4f}")
    print(f"   IoU Score:   {final_train['iou']:.4f}")
    print(f"   Precision:   {final_train['precision']:.4f}")
    print(f"   Recall:      {final_train['recall']:.4f}")
    print(f"   F1 Score:    {final_train['f1']:.4f}")
    print(f"   Accuracy:    {final_train['accuracy']:.4f}")
    print(f"   Hausdorff:   {final_train['hausdorff']:.2f}")
    
    print(f"\n🔍 VALIDACIÓN:")
    print(f"   Dice Score:  {final_val['dice']:.4f}")
    print(f"   IoU Score:   {final_val['iou']:.4f}")
    print(f"   Precision:   {final_val['precision']:.4f}")
    print(f"   Recall:      {final_val['recall']:.4f}")
    print(f"   F1 Score:    {final_val['f1']:.4f}")
    print(f"   Accuracy:    {final_val['accuracy']:.4f}")
    print(f"   Hausdorff:   {final_val['hausdorff']:.2f}")
    print("="*80)

def plot_training_history(history):
    """
    Crear gráficas completas del historial de entrenamiento
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Pérdida
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Entrenamiento', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validación', linewidth=2)
    axes[0, 0].set_title('Evolución de la Pérdida', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Pérdida')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Dice Score
    axes[0, 1].plot(epochs, history['train_metrics']['dice'], 'b-', label='Entrenamiento', linewidth=2)
    axes[0, 1].plot(epochs, history['val_metrics']['dice'], 'r-', label='Validación', linewidth=2)
    axes[0, 1].set_title('Dice Score', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # IoU Score
    axes[0, 2].plot(epochs, history['train_metrics']['iou'], 'b-', label='Entrenamiento', linewidth=2)
    axes[0, 2].plot(epochs, history['val_metrics']['iou'], 'r-', label='Validación', linewidth=2)
    axes[0, 2].set_title('IoU Score', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Época')
    axes[0, 2].set_ylabel('IoU Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Precision
    axes[0, 3].plot(epochs, history['train_metrics']['precision'], 'b-', label='Entrenamiento', linewidth=2)
    axes[0, 3].plot(epochs, history['val_metrics']['precision'], 'r-', label='Validación', linewidth=2)
    axes[0, 3].set_title('Precision', fontsize=14, fontweight='bold')
    axes[0, 3].set_xlabel('Época')
    axes[0, 3].set_ylabel('Precision')
    axes[0, 3].legend()
    axes[0, 3].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 0].plot(epochs, history['train_metrics']['recall'], 'b-', label='Entrenamiento', linewidth=2)
    axes[1, 0].plot(epochs, history['val_metrics']['recall'], 'r-', label='Validación', linewidth=2)
    axes[1, 0].set_title('Recall', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 1].plot(epochs, history['train_metrics']['f1'], 'b-', label='Entrenamiento', linewidth=2)
    axes[1, 1].plot(epochs, history['val_metrics']['f1'], 'r-', label='Validación', linewidth=2)
    axes[1, 1].set_title('F1 Score', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 2].plot(epochs, history['train_metrics']['accuracy'], 'b-', label='Entrenamiento', linewidth=2)
    axes[1, 2].plot(epochs, history['val_metrics']['accuracy'], 'r-', label='Validación', linewidth=2)
    axes[1, 2].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Época')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Hausdorff Distance
    axes[1, 3].plot(epochs, history['train_metrics']['hausdorff'], 'b-', label='Entrenamiento', linewidth=2)
    axes[1, 3].plot(epochs, history['val_metrics']['hausdorff'], 'r-', label='Validación', linewidth=2)
    axes[1, 3].set_title('Hausdorff Distance', fontsize=14, fontweight='bold')
    axes[1, 3].set_xlabel('Época')
    axes[1, 3].set_ylabel('Hausdorff Distance')
    axes[1, 3].legend()
    axes[1, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def show_predictions_binary(model, dataloader, device, num_examples: int = 3):
    model.eval()
    try:
        batch = next(iter(dataloader))
        images = batch[0].to(device)
        masks = batch[1].to(device).float()

        # Procesar máscaras para asegurar dimensiones correctas
        original_mask_shape = masks.shape
        if masks.dim() == 4 and masks.size(1) == 1:
            masks = masks.squeeze(1)
        elif masks.dim() == 3:
            pass  # Ya está en la forma correcta
        
        print(f"Debug - Forma de imágenes: {images.shape}")
        print(f"Debug - Forma original de máscaras: {original_mask_shape}")
        print(f"Debug - Forma procesada de máscaras: {masks.shape}")

        with torch.no_grad():
            outputs = model(images)
            if outputs.dim() == 4 and outputs.size(1) == 1:
                outputs = outputs.squeeze(1)
            elif outputs.dim() == 4:
                # Si el output tiene múltiples canales, tomar el primero
                outputs = outputs[:, 0, :, :]
                
            print(f"Debug - Forma de outputs: {outputs.shape}")
            
            preds = torch.sigmoid(outputs)
            preds_binary = (preds >= 0.68).float()

        # Mover todo a CPU
        images = images.cpu()
        masks = masks.cpu()
        preds = preds.cpu()
        preds_binary = preds_binary.cpu()

        for i in range(min(num_examples, images.size(0))):
            # Calcular métricas para esta imagen
            img_metrics = calculate_metrics_binary(outputs[i:i+1].cpu(), masks[i:i+1])
            
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))

            # Imagen original
            img = images[i]
            if img.shape[0] == 3:  # RGB
                # Normalizar si es necesario
                img_display = img.permute(1, 2, 0)
                if img_display.max() <= 1.0:
                    img_display = torch.clamp(img_display, 0, 1)
                else:
                    img_display = torch.clamp(img_display / 255.0, 0, 1)
                axs[0].imshow(img_display)
            elif img.shape[0] == 1:  # Escala de grises
                img_display = img.squeeze(0)
                axs[0].imshow(img_display, cmap="gray", vmin=0, vmax=1)
            else:
                # Si no es RGB ni escala de grises, usar el primer canal
                axs[0].imshow(img[0], cmap="gray")
            axs[0].set_title("Imagen original")
            axs[0].axis("off")

            # Máscara verdadera
            mask_display = masks[i]
            if mask_display.dim() > 2:
                mask_display = mask_display.squeeze()
            axs[1].imshow(mask_display, cmap="gray", vmin=0, vmax=1)
            axs[1].set_title(f"Máscara verdadera")
            axs[1].axis("off")

            # Predicción (probabilidades)
            pred_display = preds[i]
            if pred_display.dim() > 2:
                pred_display = pred_display.squeeze()
            axs[2].imshow(pred_display, cmap="gray", vmin=0, vmax=1)
            axs[2].set_title(f"Predicción (probabilidades)")
            axs[2].axis("off")

            # Predicción binaria
            pred_binary_display = preds_binary[i]
            if pred_binary_display.dim() > 2:
                pred_binary_display = pred_binary_display.squeeze()
            axs[3].imshow(pred_binary_display, cmap="gray", vmin=0, vmax=1)
            axs[3].set_title(f"Predicción binaria\nDice: {img_metrics['dice']:.3f} | IoU: {img_metrics['iou']:.3f}\nHD: {img_metrics['hausdorff']:.1f}")
            axs[3].axis("off")

            plt.tight_layout()
            plt.show()
            
            # Información adicional para debug
            print(f"Imagen {i+1}:")
            print(f"  - Máscara real: min={mask_display.min():.3f}, max={mask_display.max():.3f}")
            print(f"  - Predicción prob: min={pred_display.min():.3f}, max={pred_display.max():.3f}")
            print(f"  - Predicción binaria: min={pred_binary_display.min():.3f}, max={pred_binary_display.max():.3f}")
            print(f"  - Píxeles positivos reales: {mask_display.sum().item()}")
            print(f"  - Píxeles positivos predichos: {pred_binary_display.sum().item()}")

    except Exception as e:
        print(f"Error al visualizar predicciones: {str(e)}")
        import traceback
        traceback.print_exc()