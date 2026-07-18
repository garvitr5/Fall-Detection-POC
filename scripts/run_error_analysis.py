import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# Config
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VAL_DIR = 'data/processed/val'
RESULTS_DIR = 'results/error_analysis'
MODEL_PATHS = [
    'models/final_resnet50.keras',
    'models/resnet50_fall_detection.keras',
    'models/best_resnet50.keras'
]

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load model (try several names)
model = None
for p in MODEL_PATHS:
    if os.path.exists(p):
        print('Loading model:', p)
        model = tf.keras.models.load_model(p)
        break
if model is None:
    raise FileNotFoundError('Could not find model file in: ' + str(MODEL_PATHS))

# Data generator
datagen = ImageDataGenerator(rescale=1.0/255)
val_gen = datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Predict
steps = int(np.ceil(val_gen.samples / BATCH_SIZE))
print('Predicting...')
preds = model.predict(val_gen, steps=steps, verbose=1)
if preds.ndim > 1 and preds.shape[1] > 1:
    y_pred = np.argmax(preds, axis=1)
else:
    y_pred = (preds.ravel() > 0.5).astype(int)
y_true = val_gen.classes
class_indices = val_gen.class_indices
inv_class_indices = {v: k for k, v in class_indices.items()}
print('Classes:', class_indices)
print('Samples:', val_gen.samples)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[inv_class_indices[i] for i in range(len(inv_class_indices))],
            yticklabels=[inv_class_indices[i] for i in range(len(inv_class_indices))])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, bbox_inches='tight')
plt.close()
print('Saved confusion matrix to', cm_path)

# Classification report
report = classification_report(y_true, y_pred, target_names=[inv_class_indices[i] for i in range(len(inv_class_indices))])
report_path = os.path.join(RESULTS_DIR, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write(report)
print('Saved classification report to', report_path)

# Filepaths
filepaths = getattr(val_gen, 'filepaths', None)
if filepaths is None:
    filepaths = [os.path.join(VAL_DIR, f) for f in val_gen.filenames]

# Save CSV of predictions
results_df = pd.DataFrame({
    'filepath': filepaths,
    'true': [inv_class_indices[c] for c in y_true],
    'pred': [inv_class_indices[p] for p in y_pred]
})
results_df['correct'] = results_df['true'] == results_df['pred']
csv_path = os.path.join(RESULTS_DIR, 'validation_predictions.csv')
results_df.to_csv(csv_path, index=False)
print('Saved CSV to', csv_path)

# Save some misclassified thumbnails
mis_dir = os.path.join(RESULTS_DIR, 'misclassified')
os.makedirs(mis_dir, exist_ok=True)
mis_idx = np.where(y_true != y_pred)[0]
print('Misclassified samples:', len(mis_idx))
for i, idx in enumerate(mis_idx[:50]):
    src = filepaths[idx]
    try:
        img = Image.open(src).convert('RGB')
        img.thumbnail((320, 320))
        true_label = inv_class_indices[y_true[idx]]
        pred_label = inv_class_indices[y_pred[idx]]
        base = os.path.basename(src)
        name = f"{i:03d}__true-{true_label}__pred-{pred_label}__{base}"
        out_path = os.path.join(mis_dir, name)
        img.save(out_path)
    except Exception as e:
        print('Failed saving misclassified image', src, e)
print('Saved misclassified thumbnails to', mis_dir)

print('Done')
