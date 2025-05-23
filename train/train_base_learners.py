import numpy as np
import tensorflow as tf
from models.base_mlp import build_mlp
from utils.preprocessing import load_and_preprocess
from sklearn.model_selection import train_test_split

X, y = load_and_preprocess('data/sample_dataset.csv')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

models = []
for i in range(5):
    print(f"Training base learner {i+1}...")
    model = build_mlp(input_dim=X.shape[1], hidden_layers=[64, 32], dropout_rate=0.4)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])
    model.save(f'models/mlp_{i}.h5')
    models.append(model)