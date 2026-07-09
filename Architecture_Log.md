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

## Phase 4: Triage Classification Strategy *(Pending Execution)*

### Evaluation Metric Pivot
Discarded standard **Accuracy** as the primary evaluation metric.

Since the dataset is heavily dominated by **undamaged buildings** (reflecting real disaster scenarios), the optimization objective was redefined to maximize **Recall** for the following critical classes:

- `major-damage`
- `destroyed`

This minimizes dangerous triage blind spots (false negatives).

### Experiment Design
Structured a modular Jupyter Notebook framework to evaluate multiple class-imbalance mitigation strategies, including:

- **Algorithmic Balancing**
  - `scale_pos_weight`
- **Synthetic Data Generation**
  - SMOTE
- **Decision Threshold Optimization**
  - Custom probability threshold tuning

### Triage Classification & Class Imbalance Resolution

#### Experiment 1 (Naive Baseline): Trained a vanilla XGBClassifier on the raw 2048-d vectors.

- Result: Achieved 80% global accuracy, but exhibited severe "majority class laziness." The model achieved only a 55% Recall on "Destroyed" buildings, fatally misclassifying over 1,000 flattened structures as "No Damage."

#### Experiment 2 (Algorithmic Balancing): Engineered a dynamic sample_weight array (compute_sample_weight) to mathematically penalize the XGBoost loss function for missing minority classes.

- Result: Successfully manipulated the model's objective. Overall accuracy dropped to 72%, but "Destroyed" Recall surged to 74%, and "Major Damage" Recall more than doubled to 52%.

- Architectural Decision: Validated the trade-off. In disaster triage operations, False Positives (wasted drone flight time) are infinitely preferable to False Negatives (ignored collapsed structures).

#### Experiment 3 (Synthetic Data Generation - SMOTE): Attempted to balance the dataset by generating synthetic minority class samples using SMOTE across the 2048-dimensional feature space.

- Result: Failed to outperform algorithmic weighting. "Destroyed" Recall dropped to 61% (down from 74%), and fatal misclassifications (Destroyed predicted as No Damage) more than doubled.

- Architectural Decision: Rejected SMOTE. Confirmed that generating synthetic points via linear interpolation in a high-dimensional deep learning embedding space creates invalid data points (The Curse of Dimensionality).

- Final Baseline Selection: Officially selected the Algorithmically Weighted XGBoost Model (Experiment 2) as the production baseline due to its superior operational recall and zero synthetic data contamination.

#### Experiment 4 (Hyperparameter Tuning vs. Operational Intent): Attempted GPU-accelerated hyperparameter tuning (Custom Loop to bypass Scikit-Learn memory leaks).

- Result: The tuner optimized for f1_macro, successfully raising overall accuracy and precision, but mathematically sacrificing "Destroyed" Recall (dropping it from 74% to 70%).

- Architectural Decision: Rejected the tuned model. In disaster response, prioritizing F1/Accuracy over minority-class Recall is an operational failure. Retained the untuned Weighted Model (Experiment 2).

#### Experiment 5 (Custom Threshold Optimization): Discarded standard argmax probability boundaries (50%+). Engineered custom decision logic to heavily bias the model toward minority classes (e.g., flagging Class 3 if probability > 30%).

- Result: Achieved project climax. "Destroyed" Recall reached 78%, and "Major Damage" Recall hit 62%. Fatal misclassifications (Class 3 predicted as Class 0) were reduced by over 70% from the naive baseline (1,049 $\rightarrow$ 294).

- Conclusion: Deliberately sacrificed global accuracy (down to 67%) to maximize operational triage survival rates, successfully validating the Decoupled Multimodal Pipeline architecture.

## Phase 5: Production Deployment & UI Architecture

- Challenge: Live inference (TensorFlow + XGBoost) on high-resolution satellite imagery exceeds the memory limits of standard free-tier cloud deployment platforms (e.g., Streamlit Community Cloud's 1GB limit).

- Strategy: Implemented a Decoupled Inference Architecture.

- Execution: * Developed a local orchestration script (04_showcase_prep.py) to process a mathematically curated "Golden Sample" set of 50 images.

- Extracted ground-truth Cartesian coordinates directly from xBD WKT (Well-Known Text) JSON labels.

- Executed the heavy ML pipeline locally and serialized the metadata, coordinates, and XGBoost probabilities into a highly portable deployment payload (deploy_payload.csv containing 4,880 structures).

- Result: The final Streamlit application is entirely decoupled from TensorFlow. It relies strictly on Pandas and OpenCV, guaranteeing millisecond load times and zero risk of Out-Of-Memory (OOM) crashes in production.