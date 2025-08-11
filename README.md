# Men-Women Detection using Transfer Learning & Data Augmentation

🎯 هدف پروژه: تشخیص تصاویر مرد و زن با استفاده از مدل از پیش آموزش‌دیده **VGG16** و تکنیک‌های **Transfer Learning** و **Data Augmentation** در Keras/TensorFlow.



## 🌐 لینک ها

- [دیتاست پروژه](https://www.kaggle.com/datasets/saadpd/menwomen-classification)  
  دیتاست تصاویر مرد و زن از سایت Kaggle

- [مشاهده پروژه در Google Colab](https://colab.research.google.com/drive/1URIqEEJLyPI70XGXVyKUEBLAIYxjY09U?usp=sharing)  
  اجرای آنلاین کد پروژه در محیط Google Colab

- [مشاهده پروژه در GitHub](https://github.com/eliram88/Men_women_detection)  
  سورس کد و مستندات پروژه در GitHub



## 🔧 ابزارهای استفاده‌شده

- Python (NumPy, TensorFlow, Matplotlib, Keras, OS)
- Google Colab
- Google Drive
- Kaggle Dataset
- GitHub for version control



## 📊  دیتاست

- **Source:** Men vs Women Classification dataset (Kaggle)  
- **Samples:**  
  - Training: 1,598 images  
  - Validation: 400 images  
  - Testing: 800 images  
- **Classes:**  
  - `0` → Man  
  - `1` → Woman 



## 📊 مراحل پروژه


### 🛠 Preprocessing | پیش‌پردازش

- دانلود دیتاست از Kaggle  
- ساختاردهی پوشه‌ها برای train / validation / test  
- بارگذاری داده با `image_dataset_from_directory` در سایز `(180, 180)` پیکسل  
- اعمال Data Augmentation شامل:  
  - Random Flip  
  - Random Rotation (0.1)  
  - Random Zoom (0.2)  
- نمایش نمونه تصاویر همراه با برچسب


### 🧠 Model Design | طراحی مدل

- Base Model: **VGG16** (pre-trained on ImageNet, include_top=False)  
- Layers:  
  - Global Average Pooling  
  - Dense(256)  
  - Dropout(0.5)  
  - Dense(1, activation="sigmoid")  


### ⚙ Training | آموزش
 
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


### 📈 Results | نتایج

- Test Accuracy after Transfer Learning: ~89%
- Test Accuracy after Fine-Tuning: ~91%  
- Validation Accuracy Curve: steady improvement after fine-tuning  
- Validation Loss Curve: loss decreases steadily until convergence



## 🚀 نحوه اجرا

1) Install dependencies | نصب کتابخانه‌ها
```bash
pip install tensorflow numpy matplotlib
```

2) Run Jupyter Notebook | اجرای نوت‌بوک
```bash
jupyter notebook
```
Open the file Men_women_detection.ipynb and run all cells.



## 📁 ساختار فایل‌ها
```bash
Men_women_detection/
│
├── 📁 dataset/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── 📁 notebook/
│   └── Men_women_detection           # Data analysis & model training
│
├── 📄 README.md                      # Project documentation
│
├── requirements.txt		     # Project Libraries
```



## 🧑‍💻 توسعه‌دهنده

این پروژه توسط یک علاقه‌مند به بینایی ماشین و یادگیری عمیق طراحی و اجرا شده
با هدف توسعه مهارت در Computer Vision و کار با مدل‌های از پیش آموزش‌دیده.

✨ هدف: توسعه نمونه‌کار قابل ارائه، تمرین در پروژه‌های بینایی ماشین، یادگیری پیاده‌سازی حرفه‌ای و بهینه‌سازی مدل
