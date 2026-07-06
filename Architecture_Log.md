# Architecture Log: Disaster Response Resource Allocation

---

## Phase 1: Project Initialization & Repository Hardening

### Environment
Configured a local WSL Ubuntu environment utilizing Miniconda (`tf-gpu`) to ensure high-performance GPU access for deep learning operations while maintaining strict dependency isolation.

### Payload Protection
Configured a rigid `.gitignore` strategy prior to repository initialization. This explicitly shields the **7.8 GB Kaggle xBD challenge dataset** and all generated binary artifacts (`.tfrecord`, `.npy`) from accidentally being tracked and corrupting the Git history.

### Source Control
Initialized the project locally first and linked it to a blank remote GitHub repository to prevent nested version control conflicts.

---

## Phase 2: Decoupled Multimodal Pipeline Design

### Architecture Choice
Actively rejected standard end-to-end CNN fine-tuning. Engineered a decoupled pipeline:

```text
Image → Frozen ResNet50 → 2048-D Feature Vector → XGBoost Classifier
```

### Justification: Compute Asymmetry
Simulates a real-world disaster response edge-to-cloud infrastructure:

- **Edge devices (drones):** Run lightweight forward passes to extract feature vectors.
- **Central hubs:** Perform rapid tabular triage using XGBoost.

### Justification: Hyper-Rapid Iteration
Decoupling enables instantaneous:

- Hyperparameter tuning
- Class-weight adjustments
- Model experimentation

without requiring the pipeline to re-process the massive **7.8 GB image dataset**.

---

## Phase 3: High-Performance Data Extraction (Two-Phase Strategy)

### Data Filtering
Strictly ignored:

- Pre-disaster images
- Semantic segmentation masks

to optimize processing time and focus exclusively on the post-disaster triage challenge.

### Phase 1: Metadata Parsing
Engineered a custom JSON parser to extract bounding boxes.

- Targeted **Cartesian pixel coordinates (`xy`)**
- Explicitly discarded **GPS coordinates (`lng_lat`)**

This prevents tensor shape mismatches during image cropping.

### Phase 2: I/O Optimization
Bridged unstructured image data to structured tabular data using `tf.data.Dataset`.

Avoided traditional `cv2.imread()` or `PIL.Image.open()` loops to prevent RAM saturation.

### Serialization
Passed batched image crops through a frozen **ResNet50** GPU pipeline and serialized the resulting **2048-dimensional feature vectors** directly into `.tfrecord` format.

This avoids the severe disk I/O bottleneck caused by writing thousands of individual `.npy` files.

---

# Phase 4: Triage Classification Strategy *(Pending Execution)*

## Evaluation Metric Pivot
Discarded standard **Accuracy** as the primary evaluation metric.

Since the dataset is heavily dominated by **undamaged buildings** (reflecting real disaster scenarios), the optimization objective was redefined to maximize **Recall** for the following critical classes:

- `major-damage`
- `destroyed`

This minimizes dangerous triage blind spots (false negatives).

## Experiment Design
Structured a modular Jupyter Notebook framework to evaluate multiple class-imbalance mitigation strategies, including:

- **Algorithmic Balancing**
  - `scale_pos_weight`
- **Synthetic Data Generation**
  - SMOTE
- **Decision Threshold Optimization**
  - Custom probability threshold tuning