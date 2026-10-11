# Parkinson's Disease CT Brain Images Classification

Binary classification of CT brain images to detect Parkinson's disease using transfer learning. Three pretrained CNN architectures are benchmarked and compared.

## Task

Given a CT brain scan, predict whether the patient has Parkinson's disease (binary: 0 = Non-Parkinson's, 1 = Parkinson's).

## Models Benchmarked

| Model | Backbone | Epochs | Best Val Accuracy |
|-------|----------|--------|-------------------|
| VGG16 | ImageNet pretrained | 15 | ~86.5% |
| InceptionV3 | ImageNet pretrained | 20 | ~90.6% |
| **DenseNet121** | **ImageNet pretrained** | **15** | **~96.5%** |

**DenseNet121 is the best performer**, achieving 96.5% validation accuracy by epoch 2 — consistent with its strong performance on medical imaging tasks.

## Methodology

- **Preprocessing**: Image resizing to 224×224, normalization (0–1), class distribution analysis
- **Architecture**: Pretrained backbone (frozen) + custom classification head (Dense → Sigmoid)
- **Training**: 80/20 train/validation split, batch size 32, binary cross-entropy loss, Adam optimizer
- **Evaluation**: Accuracy, Precision, Recall, F1-Score on held-out test set
- **Visualization**: Confusion matrices, training curves, class distribution plots, pixel intensity analysis

## Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

`tensorflow` · `keras` · `scikit-learn` · `numpy` · `opencv` · `matplotlib` · `seaborn`

## Project Structure

```
Parkinson-s-Disease-CT-Brain-Images-Classification/
├── Parkinson's_Disease_CT_Brain_Images_Classification.ipynb   # Full pipeline
└── README.md
```

## Getting Started

```bash
git clone https://github.com/kaltisami/Parkinson-s-Disease-CT-Brain-Images-Classification.git
cd Parkinson-s-Disease-CT-Brain-Images-Classification
pip install tensorflow scikit-learn numpy opencv-python matplotlib seaborn
```

Open the notebook in Jupyter or Google Colab and run all cells. Place the CT brain image dataset in the project directory before running.

## Related Projects

- [chest-xray-ai-diagnosis](https://github.com/kaltisami/chest-xray-ai-diagnosis) — inference-only chest X-ray analysis system (FastAPI, Grad-CAM, anatomical segmentation)
- [Stroke-Prediction-Binary-Classification](https://github.com/kaltisami/Stroke-Prediction-Binary-Classification) — stroke risk prediction from patient health data

## License

MIT
