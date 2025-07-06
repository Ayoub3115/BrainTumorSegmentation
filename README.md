# 🧠 Brain Tumor Dataset: Brain Tumor Segmentation

The third dataset used in this project is the Brain Tumor Dataset, publicly available on Kaggle and originally published by Jun Cheng on Figshare 1. This dataset contains brain MRI images labeled for the study of intracranial tumors, and it is widely used in automated classification and segmentation tasks in neuroimaging.

The images are categorized into three tumor types:
	•	Glioma
	•	Meningioma
	•	Pituitary

Each image has a corresponding binary segmentation mask that outlines the tumor region. The data is provided in JPG format with a uniform resolution of 512×512 pixels.

For this project, the task has been formulated as a binary segmentation problem, distinguishing only between tumor regions and background (non-tumorous brain tissue). This approach merges all tumor types into a single target class, aiming to assess the model’s ability to segment masses of varying shapes, sizes, and locations without requiring histological classification.

🛠️ Preprocessing

Before training, the images underwent:
	•	Resizing to match the input dimensions required by the network
	•	Intensity normalization

The morphological and anatomical diversity within this dataset makes it well-suited for evaluating model robustness in brain segmentation tasks.

⸻

📚 References


[1] Jun Cheng. Brain Tumor Dataset. Figshare. https://figshare.com/articles/dataset/brain_tumor_dataset/1512427 (2017).

⸻
