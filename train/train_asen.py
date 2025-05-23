import numpy as np
import tensorflow as tf
from models.asen import ASEN
from utils.preprocessing import load_and_preprocess
from tensorflow.keras.models import load_model

X, y = load_and_preprocess('data/sample_dataset.csv')
base_outputs = []
# Load base learner outputs
for i in range(5):
    model = load_model(f'models/mlp_{i}.h5')
    base_outputs.append(model.predict(X))

base_outputs = np.stack(base_outputs, axis=1)  # shape: (num_samples, 5, num_classes)

# Train ASEN
asen = ASEN(num_learners=5, num_classes=6)
asen.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
asen.fit(base_outputs, y, epochs=50, batch_size=32, validation_split=0.15, callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])
asen.save('models/asen_model.h5')