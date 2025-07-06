import torch
from data import CovidDataset, SegmentationDataset, BrainDataset, TejidoDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import numpy as np

def brain_dataloader(image_dir, label_dir, batch_size=32, num_workers=-1):
    # Transforms para imágenes y máscaras
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Dataset y DataLoader
    dataset = BrainDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        transform=image_transform,
        target_transform=mask_transform
    )
    total_samples = len(dataset)
    
    # Fijar semilla para reproducibilidad
    generator = torch.Generator().manual_seed(42)

    # Crear índices aleatorios para seleccionar las muestras
    indices = torch.randperm(total_samples, generator=generator)

    # Seleccionar los primeros 1200 índices (o todos si hay menos)
    max_samples = min(1200, total_samples)
    selected_indices = indices[:max_samples]

    # Dividir en train (1000) y test (200)
    train_indices = selected_indices[:1000] if len(selected_indices) >= 1000 else selected_indices[:int(0.8*len(selected_indices))]
    test_indices = selected_indices[1000:1200] if len(selected_indices) >= 1200 else selected_indices[len(train_indices):]

    # Crear subsets
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def segmentation_dataloader(img_dir, mask_dir, batch_size=32, num_workers=-1):

    # Transformaciones
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Para las máscaras, necesitamos que mantengan valores discretos
    # No usamos ToTensor() directamente porque normaliza a [0,1]
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        # No aplicamos transformación a tensor aquí, lo haremos en __getitem__
    ])
    train_img_dir = img_dir + '/train/images'
    train_mask_dir = mask_dir + '/train/masks'
    test_img_dir = img_dir + '/test/images' 
    test_mask_dir = mask_dir + '/test/masks'
    # Datasets
    train_dataset = SegmentationDataset(
        train_img_dir, 
        train_mask_dir, 
        transform=image_transform, 
        mask_transform=mask_transform
    )
    
    test_dataset = SegmentationDataset(
        test_img_dir, 
        test_mask_dir, 
        transform=image_transform, 
        mask_transform=mask_transform
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, test_loader

def covid_dataloader(img_dir, mask_dir, batch_size=32, num_workers=-1):
    # Transformaciones
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Para las máscaras, necesitamos que mantengan valores discretos
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        # No aplicamos transformación a tensor aquí, lo haremos en __getitem__
    ])
    
    # Datasets
    train_dataset = CovidDataset(
        img_dir + '/train/images', 
        mask_dir + '/train/masks', 
        transform=image_transform, 
        mask_transform=mask_transform
    )
    
    test_dataset = CovidDataset(
        img_dir + '/test/images', 
        mask_dir + '/test/masks', 
        transform=image_transform, 
        mask_transform=mask_transform
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

def tejido_dataloader(img_dir, mask_dir, batch_size=32, num_workers=-1):
    # Transforms para imágenes y máscaras
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Dataset y DataLoader
    dataset = TejidoDataset(
        image_dir=img_dir,
        label_dir=mask_dir,
        transform=image_transform,
        target_transform=mask_transform
    )
    total_samples = len(dataset)
    # Fijar semilla para reproducibilidad
    generator = torch.Generator().manual_seed(42)
    # Crear índices aleatorios para seleccionar las muestras
    indices = torch.randperm(total_samples, generator=generator)
    # Seleccionar los primeros 1200 índices (o todos si hay menos)
    max_samples = min(1200, total_samples)
    selected_indices = indices[:max_samples]
    # Dividir en train (1000) y test (200)
    train_indices = selected_indices[:1000] if len(selected_indices) >= 1000 else selected_indices[:int(0.8*len(selected_indices))]
    test_indices = selected_indices[1000:1200] if len(selected_indices) >= 1200 else selected_indices[len(train_indices):]
    # Crear subsets
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader



#if __name__ == "__main__":
    # def probar_brain():
    #     # Ejemplo de uso
    #     img_dir = '/home/ahmedbegga/Escritorio/TFG /braintumorbueno/images'
    #     mask_dir = '/home/ahmedbegga/Escritorio/TFG /braintumorbueno/masks'
        
    #     train_loader, test_loader = brain_dataloader(img_dir, mask_dir, batch_size=32, num_workers=4)
    #     for images, masks, img_paths, mask_paths in train_loader:
    #         print(f'Batch size: {images.size(0)}')
    #         print(f'Image shape: {images.shape}')
    #         print(f'Mask shape: {masks.shape}')
    #         print(f'Image paths: {img_paths}')
    #         print(f'Mask paths: {mask_paths}')
    #         break
    #     # Vamos a mostrar una cuadricula de 2x3, donde pondremos en la primera fila las imágenes y en la segunda las máscaras
        

    #     def show_images(images, masks):
    #         fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    #         for i in range(3):
    #             # Mostrar imagen
    #             img = images[i].permute(1, 2, 0).numpy()
    #             img = np.clip(img, 0, 1)  # Asegurarse
    #             axs[0, i].imshow(img)
    #             axs[0, i].axis('off')
    #             axs[0, i].set_title(f'Image {i+1}')
    #             # Mostrar máscara
    #             mask = masks[i].permute(1, 2, 0).numpy()
    #             mask = np.clip(mask, 0, 1)  # Asegurarse
    #             axs[1, i].imshow(mask)
    #             axs[1, i].axis('off')
    #             axs[1, i].set_title(f'Mask {i+1}')
    #         plt.tight_layout()
    #         plt.show()
    #     # Vamos a crear una función en la que coloree la imagen original usando la máscara, como si fuera un overlay, para ello solo usaremos una imagen de 1x3
    #     def overlay_images(images, masks):
    #         fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    #         for i in range(3):
    #             # Obtener imagen y máscara
    #             img = images[i].permute(1, 2, 0).numpy()
    #             mask = masks[i].permute(1, 2, 0).numpy()
                
    #             # Asegurarse de que los valores estén en el rango [0, 1]
    #             img = np.clip(img, 0, 1)
    #             mask = np.clip(mask, 0, 1)
                
    #             # Crear overlay
    #             overlay = img * (1 - mask) + mask * np.array([1, 0, 0])  # Rojo para la máscara
    #             axs[i].imshow(overlay)
    #             axs[i].axis('off')
    #             axs[i].set_title(f'Overlay {i+1}')
    #         plt.tight_layout()
    #         plt.show()
    #     # Mostrar imágenes con overlay
        
    #     # Mostrar las primeras 3 imágenes y sus máscaras
    #     for images, masks, img_paths, mask_paths in train_loader:
    #         show_images(images, masks)
    #         overlay_images(images, masks)
    #         break
    # def probar_covid():
    #     # Ejemplo de uso
    #     img_dir = '/home/ahmedbegga/Escritorio/TFG /coviddatabase/COVID-19_Radiography_Dataset/dataset/'
    #     mask_dir = '/home/ahmedbegga/Escritorio/TFG /coviddatabase/COVID-19_Radiography_Dataset/dataset/'
        
    #     train_loader, test_loader = covid_dataloader(img_dir, mask_dir, batch_size=32, num_workers=4)
    #     for images, masks, img_paths, mask_paths in train_loader:
    #         print(f'Batch size: {images.size(0)}')
    #         print(f'Image shape: {images.shape}')
    #         print(f'Mask shape: {masks.shape}')
    #         print(f'Image paths: {img_paths}')
    #         print(f'Mask paths: {mask_paths}')
    #         break
    #     def show_images(images, masks):
    #         fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    #         for i in range(3):
    #             # Mostrar imagen
    #             img = images[i].permute(1, 2, 0).numpy()
    #             img = np.clip(img, 0, 1)
    #             axs[0, i].imshow(img)
    #             axs[0, i].axis('off')
    #             axs[0, i].set_title(f'Image {i+1}')
    #             # Mostrar máscara
    #             # la mascara es de tamaño 224x224
    #             mask = masks[i].squeeze().numpy()
    #             axs[1, i].imshow(mask, cmap='gray')
    #             axs[1, i].axis('off')
    #             axs[1, i].set_title(f'Mask {i+1}')
    #         plt.tight_layout()
    #         plt.show()
    #     def overlay_images(images, masks):
    #         fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    #         for i in range(3):
    #             # Obtener imagen y máscara
    #             img = images[i].permute(1, 2, 0).numpy()
    #             mask = masks[i].squeeze().numpy()
                
    #             # Asegurarse de que los valores estén en el rango [0, 1]
    #             img = np.clip(img, 0, 1)
    #             mask = np.clip(mask, 0, 1)
                
    #             # Crear overlay
    #             overlay = img * (1 - mask[..., None]) + mask[..., None] * np.array([1, 0, 0])  # Rojo para la máscara
    #             axs[i].imshow(overlay)
    #             axs[i].axis('off')
    #             axs[i].set_title(f'Overlay {i+1}')
    #         plt.tight_layout()
    #         plt.show()
    #     for images, masks, img_paths, mask_paths in train_loader:
    #         show_images(images, masks)
    #         overlay_images(images, masks)
    #         break
    # def probar_tejido():
    #     img_dir = '/home/ahmedbegga/Escritorio/TFG /segmentacion/dataset/'
    #     mask_dir = '/home/ahmedbegga/Escritorio/TFG /tejidos/EBHI-SEG/dataset_final/label/'

    #     train_loader, test_loader = tejido_dataloader(img_dir, mask_dir, batch_size=32, num_workers=4)
    #     for images, masks, img_paths, mask_paths in train_loader:
    #         print(f'Batch size: {images.size(0)}')
    #         print(f'Image shape: {images.shape}')
    #         print(f'Mask shape: {masks.shape}')
    #         print(f'Image paths: {img_paths}')
    #         print(f'Mask paths: {mask_paths}')
    #         break
    #     def show_images(images, masks):
    #         fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    #         for i in range(3):
    #             # Mostrar imagen
    #             img = images[i].permute(1, 2, 0).numpy()
    #             img = np.clip(img, 0, 1)
    #             axs[0, i].imshow(img)
    #             axs[0, i].axis('off')
    #             axs[0, i].set_title(f'Image {i+1}')
    #             # Mostrar máscara
    #             mask = masks[i].squeeze().numpy()
    #             axs[1, i].imshow(mask, cmap='gray')
    #             axs[1, i].axis('off')
    #             axs[1, i].set_title(f'Mask {i+1}')
    #         plt.tight_layout()
    #         plt.show()
    #     def overlay_images(images, masks):
    #         fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    #         for i in range(3):
    #             # Obtener imagen y máscara
    #             img = images[i].permute(1, 2, 0).numpy()
    #             mask = masks[i].squeeze().numpy()
                
    #             # Asegurarse de que los valores estén en el rango [0, 1]
    #             img = np.clip(img, 0, 1)
    #             mask = np.clip(mask, 0, 1)
                
    #             # Crear overlay
    #             overlay = img * (1 - mask[..., None]) + mask[..., None] * np.array([1, 0, 0])  # Rojo para la máscara
    #             axs[i].imshow(overlay)
    #             axs[i].axis('off')
    #             axs[i].set_title(f'Overlay {i+1}')
    #         plt.tight_layout()
    #         plt.show()
    #     for images, masks, img_paths, mask_paths in train_loader:
    #         show_images(images, masks)
    #         overlay_images(images, masks)
    #         break
    # def probar_segmentation():
    #     img_dir = '/home/ahmedbegga/Escritorio/TFG /segmentacion/dataset/'
    #     mask_dir = '/home/ahmedbegga/Escritorio/TFG /segmentacion/dataset/'

    #     train_loader, test_loader = segmentation_dataloader(img_dir, mask_dir, batch_size=32, num_workers=8)
    #     for images, masks, img_paths, mask_paths in train_loader:
    #         print(f'Batch size: {images.size(0)}')
    #         print(f'Image shape: {images.shape}')
    #         print(f'Mask shape: {masks.shape}')
    #         print(f'Image paths: {img_paths}')
    #         print(f'Mask paths: {mask_paths}')
    #         break
    #     def show_images(images, masks):
    #         fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    #         for i in range(3):
    #             # Mostrar imagen
    #             img = images[i].permute(1, 2, 0).numpy()
    #             img = np.clip(img, 0, 1)
    #             axs[0, i].imshow(img)
    #             axs[0, i].axis('off')
    #             axs[0, i].set_title(f'Image {i+1}')
    #             # Mostrar máscara
    #             mask = masks[i].squeeze().numpy()
    #             axs[1, i].imshow(mask, cmap='gray')
    #             axs[1, i].axis('off')
    #             axs[1, i].set_title(f'Mask {i+1}')
    #         plt.tight_layout()
    #         plt.show()
    #     def overlay_images(images, masks):
    #         fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    #         for i in range(3):
    #             # Obtener imagen y máscara
    #             img = images[i].permute(1, 2, 0).numpy()
    #             mask = masks[i].squeeze().numpy()
                
    #             # Asegurarse de que los valores estén en el rango [0, 1]
    #             img = np.clip(img, 0, 1)
    #             mask = np.clip(mask, 0, 1)
                
    #             # Crear overlay
    #             overlay = img * (1 - mask[..., None]) + mask[..., None] * np.array([1, 0, 0])
    #             axs[i].imshow(overlay)
    #             axs[i].axis('off')
    #             axs[i].set_title(f'Overlay {i+1}')
    #         plt.tight_layout()
    #         plt.show()
    #     for images, masks, img_paths, mask_paths in train_loader:
    #         show_images(images, masks)
    #         overlay_images(images, masks)
    #         break
    #probar_segmentation()
    #probar_tejido()
    #probar_covid()
    #probar_brain()
