import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from utils.preprocessing import load_and_preprocess
from utils.metrics import compute_metrics
from utils.visualization import plot_confusion_matrix

X, y = load_and_preprocess('data/sample_dataset.csv')
y_true = np.argmax(y, axis=1)

asen_model = load_model('models/asen_model.h5')
base_outputs = []
for i in range(5):
    base_model = load_model(f'models/mlp_{i}.h5')
    base_outputs.append(base_model.predict(X))
base_outputs = np.stack(base_outputs, axis=1)

preds = asen_model.predict(base_outputs)
y_pred = np.argmax(preds, axis=1)

labels = ['Wheat', 'Maize', 'Sugarcane', 'Mustard', 'Cotton', 'Potato']

metrics = compute_metrics(y_true, y_pred)
print("Evaluation Metrics:")
for k, v in metrics.items():
    print(f"{k.capitalize()}: {v:.4f}")

plot_confusion_matrix(y_true, y_pred, labels)