# 🧠 Brain Tumor Dataset: Brain Tumor Segmentation

The third dataset used in this project is the **Brain Tumor Dataset**, publicly available on Kaggle and originally published by **Jun Cheng**. This dataset contains brain MRI images labeled for the study of **intracranial tumors**, and it is widely used in automated classification and segmentation tasks in neuroimaging.

---

## 📂 Dataset Description

The MRI images are categorized into three tumor types:
- **Glioma**
- **Meningioma**
- **Pituitary**

Each image has an associated **binary segmentation mask** that outlines the tumor region. All images are provided in **JPG format** with a uniform resolution of **512×512 pixels**.

For the purposes of this project, the task is formulated as a **binary segmentation problem**, distinguishing between **tumor regions** and **non-tumorous brain tissue**. All tumor types are merged into a **single target class**, focusing on the model's ability to **segment tumors of varying shapes, sizes, and locations**, rather than performing subtype classification.

---

## 🛠️ Preprocessing

Prior to training, the following preprocessing steps were applied:
- **Resizing** all images to match the input dimensions required by the neural networks
- **Intensity normalization** to ensure consistent contrast across scans

The **morphological and anatomical diversity** in this dataset makes it well-suited for evaluating **model robustness** in brain tumor segmentation tasks.

---

## 🧪 Models Used

Three different deep learning architectures were implemented and compared to evaluate segmentation performance:

1. **U-Net**
   - The classic encoder-decoder architecture designed for biomedical image segmentation.
   - Captures fine-grained localization through skip connections.

2. **U-Net++**
   - An enhanced version of U-Net with nested and dense skip pathways.
   - Improves segmentation accuracy by refining feature aggregation.

3. **Swin-Unet** (Innovative Transformer-Based Architecture)
   - A novel architecture combining **Swin Transformers** with U-Net design principles.
   - Utilizes **pretrained weights** to leverage contextual understanding from large-scale vision tasks.
   - Demonstrates strong performance in capturing global and local features for complex segmentation problems.

These models were evaluated to determine their effectiveness in **binary brain tumor segmentation**, considering **generalization**, **precision**, and **boundary accuracy**.

---

## 📚 References

[1] Jun Cheng. *Brain Tumor Dataset*. Figshare. https://figshare.com/articles/dataset/brain_tumor_dataset/1512427 (2017).

---
