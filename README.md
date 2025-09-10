# 🖼️ Image Classification on CIFAR-10: From Scratch to Transfer Learning 🚀  
*How far can we push accuracy on CIFAR-10 by evolving from a simple CNN to cutting-edge pretrained models?*  

---

## 📊 Dataset Description  
- **Dataset**: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)  
- **Size**: 60,000 color images (32x32 pixels) across **10 balanced classes**:  
  ✈️ airplane | 🚗 automobile | 🐦 bird | 🐱 cat | 🦌 deer | 🐶 dog | 🐸 frog | 🐴 horse | 🚢 ship | 🚚 truck  
- **Split**: 50,000 training images + 10,000 testing images.  
- **Challenges**:  
  - Low-resolution images → harder to extract strong features.  
  - Some categories are very similar (cat vs dog, car vs truck).  

---

## 🎯 Research Goal  
This project set out to answer a key question:  

👉 *What happens when we compare custom-built CNNs against modern transfer learning models on CIFAR-10?*  

Unlike many projects that stop at a single CNN, this work explores the **evolution of improvements** step by step:  

1. **Baseline CNN (from scratch)** — a simple ConvNet to establish a starting point.  
2. **Deeper CNN** — added more convolutional layers and dropout to extract richer features.  
3. **BatchNorm, Dropout & Regularization** — introduced Batch Normalization, GlobalAveragePooling and L2 weight decay to stabilize and regularize training.  
4. **Early Stopping** — tuned training with early stopping to avoid overfitting.  
5. **Data Augmentation** — applied random rotations, flips and shifts to improve robustness and generalization.  
6. **Transfer Learning (MobileNetV2 variants)** — benchmarked several MobileNetV2 head/fine-tuning strategies to leverage pretrained ImageNet features. 

---

## 🛠️ Methodology & Models  

### 🔹 Custom CNNs  
- **Baseline CNN**  
  - Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Dense(128) → Softmax  
  - **Test Accuracy**: ~68.9%  

- **Deeper CNN + Dropout**  
  - Added Conv2D(128), Dense(256), Dropout(0.5)  
  - **Test Accuracy**: **73.4%** | Train Acc: 85%  

- **CNN + BatchNorm + GAP**  
  - Added BatchNormalization + GlobalAveragePooling  
  - **Test Accuracy**: **73.1%** | Train Acc: 91%  

- **CNN + Regularizer (L2)**  
  - Added weight decay regularization  
  - **Test Accuracy**: **72.5%** | Train Acc: 93%  

- **CNN + Early Stopping**  
  - Hyperparameter tuning + early stopping  
  - **Test Accuracy**: **72.9%** | Train Acc: 94%  

- **CNN + Data Augmentation**  
  - Rotations, flips, shifts  
  - **Test Accuracy**: **76.9%** ✅ | Train Acc: 74%  

---

### 🔹 Transfer Learning  

#### MobileNetV2 Experiments  
- **MobileNetV2.1** → GAP → Dense(128, ReLU) → Dense(10, Softmax)  
  - Training: 5 epochs, batch size 64  
  - **Test Accuracy**: **86.6%** ✅  

- **MobileNetV2.2** → GAP → Dropout(0.3) → Dense(10, Softmax)  
  - Training: Frozen base, 10 epochs, default batch size  
  - **Test Accuracy**: **85.6%**  

- **MobileNetV2.3** → Same as V2.2, batch size 64  
  - Training: Frozen base, 10 epochs  
  - **Test Accuracy**: **85.6%**    

---

## 📈 Results Overview  

| Model Version                       | Test Accuracy | Training Accuracy |
|-------------------------------------|---------------|------------------|
| Baseline CNN (2 Conv layers)        | ~68.9%        | ~84%             |
| Deeper CNN (3 Conv + Dropout)       | **73.4%**     | 85%              |
| CNN + BatchNorm + GAP               | **73.1%**     | 91%              |
| CNN + Regularizer (L2)              | **72.5%**     | 93%              |
| CNN + Early Stopping                | **72.9%**     | 94%              |
| CNN + Data Augmentation             | **76.9%** ✅  | 74%              |
|-------------------------------------|---------------|------------------|
| MobileNetV2.1 (Dense Head)          | **86.6%** ✅  |                 |
| MobileNetV2.2 (Dropout Head)        | **85.6%**     | -               |
| MobileNetV2.3 (Dropout + batch=64)  | **85.6%**     | -                |

---

## 💡 Key Insights  

- 📈 Custom CNNs improved step by step, but plateaued around ~77%.  
- 🧠 Transfer learning (MobileNetV2) **significantly outperformed scratch CNNs**.  
- 🔍 **MobileNetV2.1** achieved state-of-the-art performance on CIFAR-10 at **86.6%**.  
- 🐱🐶 Most misclassifications: **cats vs dogs**, **cars vs trucks**.  

---

## ⚙️ How to Reproduce  

1.**Clone the repo**  
```bash
git clone https://nataliadominguez99/Image-classification-using-CNN-and-Transfer-Learning.git
cd your-repo
```
2. **Navigate to the project folder**
   ```bash
    cd Image-classification-using-CNN-and-Transfer-Learning

3. **Open the Jupyter Notebook**
- If you use Jupyter Notebook:
   ```bash
   jupyter notebook "ProjectCifar.ipynb"
- Or, open it in VSCode by double-clicking the file or using:
   ```bash
    code "ProjectCifar.ipynb"

4. **Run all cells**
- Select Cell > Run All in Jupyter Notebook or VSCode to reproduce the analysis.
---

## Install dependencies

```bash
pip install -r requirements.txt
```
---

## Run the notebook  

Open `ProjectCifar.ipynb` in Jupyter Notebook and run all cells.  

- **Python version**: 3.10+  
- **Key Libraries**: TensorFlow/Keras, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

---

## 🚀 Next Steps  

- Extend to **CIFAR-100** (100 classes, more fine-grained).  
- Try larger **transfer learning models** (ResNet50, EfficientNetB7).  
- Deploy as a **Streamlit/Flask web app** for interactive demos.  
- Experiment with **ensembling multiple CNNs + transfer models**.

---

## 📂 Repository Structure
```bash
├── ProjectCifar.ipynb # Jupyter Notebook with full pipeline
├── requirements.txt   # Dependencies for reproduction
├── Image classification using CNN and Transfer Learning.pptx  # Project presentation slides
└── README.md          # Project documentation
```
