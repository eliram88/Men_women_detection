# Men-Women Detection using Transfer Learning & Data Augmentation

🎯 Project Goal: Classify images of men and women using a pre-trained VGG16 model with Transfer Learning and Data Augmentation techniques in Keras/TensorFlow.



## 🌐 Links

- [Dataset](https://www.kaggle.com/datasets/saadpd/menwomen-classification)  
  Men vs Women Classification Dataset (Kaggle)

- [View project in Google Colab](https://colab.research.google.com/drive/1URIqEEJLyPI70XGXVyKUEBLAIYxjY09U?usp=sharing)  
  Google Colab Notebook: Run the project online

- [View project in GitHub](https://github.com/eliram88/Men_women_detection)  
  GitHub Repository: Source code & documentation



## 🔧 Tools & Libraries

- Python (NumPy, TensorFlow, Matplotlib, Keras, OS)
- Google Colab
- Google Drive
- Kaggle Dataset
- GitHub for version control



## 📊  Dataset

- **Source:** Men vs Women Classification dataset (Kaggle)  
- **Samples:**  
  - Training: 1,598 images  
  - Validation: 400 images  
  - Testing: 800 images  
- **Classes:**  
  - `0` → Man  
  - `1` → Woman 



## 📊 Project Stages


### 🛠 Preprocessing

- Downloaded dataset from Kaggle 
- Structured directories into train / validation / test 
- Loaded images using image_dataset_from_directory with size (180, 180) 
- Applied Data Augmentation:  
  - Random Flip  
  - Random Rotation (0.1)  
  - Random Zoom (0.2)  
- Displayed sample images with labels


### 🧠 Model Design

- Base Model: **VGG16** (pre-trained on ImageNet, include_top=False)  
- Layers:  
  - Global Average Pooling  
  - Dense(256)  
  - Dropout(0.5)  
  - Dense(1, activation="sigmoid")  


### ⚙ Training
 
**Phase 1 — Transfer Learning**  
- Base model frozen  
- Optimizer: Adam  
- Loss: Binary Crossentropy  
- Metrics: Accuracy  
- Epochs: 30  
- Callback: ModelCheckpoint (best val_loss)

**Phase 2 — Fine-Tuning**  
- Unfreeze last 4 layers of VGG16  
- Reduce learning rate → `1e-5`  
- Epochs: 20  
- Callback: ModelCheckpoint (best val_loss) 


### 📈 Results

- Test Accuracy after Transfer Learning: ~89%
- Test Accuracy after Fine-Tuning: ~91%  
- Validation Accuracy Curve: steady improvement after fine-tuning  
- Validation Loss Curve: loss decreases steadily until convergence



## 🚀 How to Run

1) Install dependencies
```bash
pip install tensorflow numpy matplotlib
```

2) Run Jupyter Notebook
```bash
jupyter notebook
```
Open the file Men_women_detection.ipynb and run all cells.



## 📁 Project Structure
```bash
Men_women_detection/
│
├── 📁 dataset/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── 📁 notebook/
│   └── Men_women_detection        # Data analysis & model training
│
├── 📄 README.md                   # Project documentation
│
├── requirements.txt		         # Project Libraries
```



## 🧑‍💻 Developer

This project was developed by a computer vision and deep learning enthusiast with the goal of:

- Enhancing skills in image classification & transfer learning
- Building a professional portfolio project
- Practicing model optimization for real-world datasets
