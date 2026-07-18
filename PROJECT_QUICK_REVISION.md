# Human Fall Detection - Quick Interview Revision

Use this immediately before an interview. For evidence and full explanations, see `COMPLETE_PROJECT_INTERVIEW_GUIDE.md`.

## One-Line Summary

An image-based Fall/No-Fall proof of concept comparing a Custom CNN, MobileNetV3, and ResNet50, with error analysis, calibration, Grad-CAM, and a dark/light Streamlit dashboard.

## 30-Second Pitch

“I combined three fall/activity image sources into 1,123 training and 361 validation images. I trained a custom CNN and transfer-learned MobileNetV3 and ResNet50 models with augmentation, class weights, checkpointing, and fine-tuning. The saved tuned ResNet run had the strongest validation result at 92.8% accuracy and .985 PR-AUC. I also diagnosed a label-direction bug, tested temperature scaling, analyzed errors with Grad-CAM, and deployed a Streamlit POC. The key caveats are no independent test set and a verified mismatch between the app binary and tuned headline metrics.”

## Exact Contract

- Input: one RGB image, resized to `224x224`.
- Labels: `No Fall=0`, `Fall=1`.
- Output: sigmoid scalar interpreted as `P(Fall)`.
- Default decision: Fall when probability is at least `0.50`.
- Scope: frame classifier POC, not a temporal detector or emergency service.

## Dataset Numbers

| Split | No Fall | Fall | Total |
|---|---:|---:|---:|
| Train | 684 | 439 | 1,123 |
| Validation | 215 | 146 | 361 |

Sources: Fall Dataset; sampled falldataset.com sequences; selected HAR negatives such as sleeping, sitting, laptop use, running, eating, and texting.

No independent test set is present. Subject/video/site separation is not verified. Earlier duplicate analysis flagged 41 perceptual-hash candidates; the exact record for the apparent one-image cleanup is incomplete.

## Training Recipe

- Augmentation: horizontal flip, small rotation `.05`, zoom `.10`, contrast `.10`.
- Class weights: No Fall `.8209`, Fall `1.2790`.
- Loss/optimizer: binary cross-entropy and Adam.
- Metrics: accuracy, precision, recall, ROC-AUC, PR-AUC.
- Tuned checkpoint: maximum validation PR-AUC.
- Early stopping: patience 5, restore best weights.
- LR reduction: validation loss, factor `.2`, patience 2.
- Head LR: `.001`; fine-tuning LR: `1e-5`.
- Batch size: 32 for CNN/MobileNet; 16 for ResNet.
- Fine-tuning: nominal last 30 backbone layers; BatchNorm frozen.

## Architectures

**Custom CNN:** four Conv-BN-ReLU-MaxPool blocks (32/64/128/256), global average pooling, Dense-128, dropout, sigmoid.

**MobileNetV3Small:** ImageNet backbone with built-in preprocessing, GAP, Dense-128 with L2, dropout `.3`, sigmoid. Best efficiency option.

**ResNet50:** ImageNet residual backbone, preprocessing, GAP, Dense-256 with L2, dropout `.4`, sigmoid. Best saved validation ranking, but largest/slower.

## Results to Memorize

| Model | Stage | Accuracy | Precision | Recall | PR-AUC | TN/FP/FN/TP |
|---|---|---:|---:|---:|---:|---|
| Custom CNN | baseline | 57.06% | 38.46% | 10.27% | .3816 | 191/24/131/15 |
| MobileNetV3 | baseline | 69.81% | 58.77% | 84.93% | .7165 | 128/87/22/124 |
| ResNet50 | baseline | 88.09% | 78.45% | 97.26% | .9659 | 176/39/4/142 |
| Custom CNN | tuned | 58.45% | 48.78% | 54.79% | .4741 | 131/84/66/80 |
| MobileNetV3 | tuned | 78.12% | 67.36% | 89.04% | .8401 | 152/63/16/130 |
| ResNet50 | tuned | **92.80%** | **97.62%** | **84.25%** | **.9851** | **212/3/23/123** |

Baseline local inference: CNN about 6.81 ms/image and 4.93 MB; MobileNet 6.97 ms and 4.15 MB; ResNet 33.96 ms and 90.65 MB. Treat these as one-machine measurements, not universal latency.

## Safety Trade-off

Baseline ResNet caught 142/146 Falls but raised 39 false alarms. Tuned ResNet produced only 3 false alarms but missed 23 Falls at threshold .50. “Best” therefore depends on miss cost, false-alert tolerance, latency, and threshold. A production threshold must be selected on untouched grouped data against a written safety objective.

## Metrics in One Breath

- Precision `TP/(TP+FP)`: how trustworthy Fall alerts are.
- Recall `TP/(TP+FN)`: how many real Falls are caught.
- Specificity `TN/(TN+FP)`: how many No-Fall cases are cleared.
- F1: harmonic balance of precision and recall.
- ROC-AUC: positive-versus-negative ranking over thresholds.
- PR-AUC: positive ranking/alert trade-off; especially relevant here.
- Log-loss: probability quality, harsh on confident mistakes.
- Accuracy: useful but insufficient because it hides error type.

## Label-Direction Bug

Directory inference returned `fall=0`, `no_fall=1`, but the model had been trained with explicit `no_fall=0`, `fall=1`. The wrong evaluation inverted the model score and produced AUC near `.0098`.

Correct logic:

```python
y_true = 1 - directory_labels
fall_probability = model_output  # already P(Fall)
```

Lesson: label indices and output semantics are separate contracts. Serialize the map and test known examples end to end.

## Calibration

- Method: temperature scaling, `sigmoid(logit(p)/T)`.
- Split: 180 calibration and 181 evaluation samples, both derived from validation.
- Supported optimum: `T=1.06`, not the stale summary value `5.0`.
- Result: evaluation RMSE and log-loss slightly worsened; do **not** claim calibration improved the model.
- AUC stays unchanged because positive temperature scaling preserves ranking.

## Grad-CAM

Steps: gradients of target score to last conv maps; average gradients per channel; weighted sum plus ReLU; normalize/resize/overlay.

Critical limitation: the notebook loads baseline-style `final_resnet50.keras`, not tuned ResNet. Its manually reconstructed forward score also differed from `model.predict`, so the current heatmaps are exploratory, not faithful explanations of the tuned/deployed claim.

## Streamlit and Deployment

- `streamlit_app/app.py`: 689 lines, 15 dashboard pages.
- Inputs: upload and camera.
- `st.cache_resource`: Keras model; `st.cache_data`: CSV/history artifacts.
- Theme: session-state dark/light CSS variables.
- App artifact: `streamlit_app/model/final_resnet50.keras`.
- Cloud failure cause: Python 3.14 had no compatible TensorFlow wheel.
- Deployment runtime: use Python 3.12 with compatible pinned requirements.
- Git LFS stores the large model binary; Git stores its pointer.

## Four Inconsistencies to Say Honestly

1. App loads a verified baseline-style model while dashboard panels present tuned metrics.
2. `temperature_summary.csv` reports `T=5.0`, conflicting with the search/notebook optimum `1.06`.
3. Grad-CAM analyzes the baseline-style artifact and may not reproduce its exact forward preprocessing.
4. Old notebook outputs preserve the early reversed-label evaluation alongside corrected analysis.

## Five Highest-Priority Fixes

1. Choose one champion and record its hash, input/class contract, threshold, and metrics.
2. Generate app predictions, evaluation, calibration, error tables, and Grad-CAM from that exact artifact/shared inference function.
3. Create subject/video/site-grouped train, calibration, and untouched test manifests.
4. Add mapping, preprocessing, artifact-hash, known-prediction, clean-LFS-load, and UI smoke tests.
5. Tune threshold under an explicit miss/false-alert cost and report slice metrics/confidence intervals.

## Production Redesign

Person detection/tracking -> RGB or pose clip features -> temporal classifier -> persistence/hysteresis -> uncertainty/abstention -> human-confirmed alert. Measure event sensitivity, false alerts per camera-hour, time-to-detect, target-device latency, drift, and slice performance.

## Fast Interview Answers

**Why ResNet?** Strong reusable residual features; best validation ranking here, with compute cost.

**Why MobileNet?** Much smaller/faster and still good recall; suitable for edge constraints.

**Why class weights?** Increase the minority Fall contribution without duplicating images.

**Why PR-AUC?** It focuses on Fall-positive performance and reflects precision/recall across thresholds.

**Why freeze BatchNorm?** Small-batch fine-tuning can destabilize its moving statistics.

**Why not claim production-ready?** No independent/grouped test, frame-only input, small mixed-source data, artifact mismatch, and no real-world event validation.

**Biggest learning?** Architecture matters, but explicit data/label contracts and artifact lineage decide whether metrics actually describe the deployed system.

## Claims to Avoid

- “The model is 92.8% accurate in the real world.”
- “Calibration improved it.”
- “Grad-CAM proves the model looks at the person.”
- “There is no leakage.”
- “The Streamlit app deploys the tuned model.”
- “A fixed .50 threshold is safe.”
- “This is a medical-grade fall detector.”

## Strong Closing Line

“The project demonstrates a complete ML workflow and strong validation ranking, but my audit found that trustworthy deployment requires one versioned champion, grouped independent evaluation, shared inference logic, and safety-driven threshold validation.”
