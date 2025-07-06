import os
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image

class CovidDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None):
        """
        Dataset para segmentación con una sola clase por imagen.
        
        Args:
            img_dir: Directorio de imágenes
            mask_dir: Directorio de máscaras (binarias)
            transform: Transformaciones para imágenes
            mask_transform: Transformaciones para máscaras
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Obtener nombres de archivos
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

        # Extraer clases únicas del nombre de archivo
        self.classes = sorted(set(f.split('-')[0] for f in self.img_files))
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes)}

        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        mask_name = self.mask_files[idx]

        # Cargar imagen
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')

        # Cargar máscara (blanco y negro)
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path).convert('L')  # L = escala de grises

        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Convertimos a array y binarizamos la máscara
        mask_array = np.array(mask)
        mask_binary = (mask_array > 127).astype(np.uint8)  # 1 si blanco, 0 si negro
        mask_tensor = torch.from_numpy(mask_binary).long()

        # Obtener la clase desde el nombre del archivo
        class_name = img_name.split('-')[0]
        class_idx = self.class_to_index[class_name]

        return img, mask_tensor, img_name, mask_name
    

    def get_original_item(self, idx):
        """Devuelve las imágenes originales (no transformadas) para visualización"""
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        img = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('RGB'))  # Cambiar a RGB para trabajar con colores
        
        return img, mask, img_path, mask_path

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None):
        """
        Dataset para segmentación de imágenes médicas.
        
        Args:
            img_dir: Directorio donde se encuentran las imágenes
            mask_dir: Directorio donde se encuentran las máscaras
            transform: Transformaciones para las imágenes
            mask_transform: Transformaciones para las máscaras
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Obtener nombres de archivos
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Cargar imagen
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = Image.open(img_path).convert('RGB') 
        
        # Cargar máscara
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        mask = Image.open(mask_path).convert('RGB')  # Cambiar a RGB para trabajar con colores
        
        # Aplicar transformaciones a la imagen
        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)
        # Aplicar transformaciones a la máscara (redimensionamiento)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        # Convertir máscara a numpy array
        mask_array = np.array(mask)
        
        # Colores de referencia para las clases
        reference_colors = np.array([
            [0, 0, 0],      # Clase 0: Fondo (negro)
            [0, 0, 255],    # Clase 1: Azul
            [0, 255, 0],    # Clase 2: Verde
            [255, 0, 0]     # Clase 3: Rojo
        ])
        
        # Aplanar la máscara para procesamiento eficiente
        height, width = mask_array.shape[:2]
        pixels = mask_array.reshape(-1, 3)
        
        # Pre-calcular las distancias para todos los píxeles a todos los colores de referencia
        # Redimensionar para broadcasting: (n_pixels, 1, 3) - (1, n_classes, 3)
        diff = pixels[:, np.newaxis, :] - reference_colors[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))  # Distancia euclidiana
        
        # Para cada píxel, encuentra la clase con menor distancia
        closest_classes = np.argmin(distances, axis=1)
        
        # Reshape de vuelta a la forma original
        mask_mapped = closest_classes.reshape(height, width)
        
        # Convertir a tensor de tipo long
        mask_tensor = torch.from_numpy(mask_mapped.astype(np.int64))
        
        return img, mask_tensor, self.img_files[idx], self.mask_files[idx]

    def get_original_item(self, idx):
        """Devuelve las imágenes originales (no transformadas) para visualización"""
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        img = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('RGB'))  # Cambiar a RGB para trabajar con colores
        
        return img, mask, img_path, mask_path

class BrainDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_names = sorted(os.listdir(image_dir))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        label_path = os.path.join(self.label_dir, self.image_names[idx])

        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("L")  # Máscaras en escala de grises

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label, img_path, label_path

class TejidoDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_names = sorted(os.listdir(image_dir))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        label_path = os.path.join(self.label_dir, self.image_names[idx])

        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("L")  # Máscaras en escala de grises

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label, img_path, label_path