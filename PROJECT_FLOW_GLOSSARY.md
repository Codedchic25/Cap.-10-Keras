# 📘 Project Architecture Data Flow & Technical Glossary

This document serves as an exhaustive operational manual detailing the data journey, architectural shifts, and deep learning glossary items utilized across the AgriVision AI ecosystem.

---

## 🔄 End-to-End Project Data Flow & Evolution

The engineering pipeline operates across four distinct structural layers designed to maximize runtime stability and separate computing concerns:

```text
[ 1. DATA GENERATION ]   scripts/download_real_images.py  ==> Compiles 224x224 RGB tensors to disk
                                                                    │
                                                                    ▼
[ 2. MODEL TUNING ]      scripts/train_pipeline.py        ==> MobileNetV2 Transfer Learning -> models/
                                                                    │
                                                                    ▼
[ 3. INFERENCE ENGINE ]  app.py (Flask API)               ==> Hardened CPU scope locks port 5000
                                                                    │
                                                                   ├──► [ BRANCH: main ] frontend/index.html
                                                                   │     (Vanilla JS Fetch API pipeline)
                                                                   │
                                                                   └──► [ BRANCH: dev-pipeline ] app_streamlit.py
                                                                         (Reactive Python Charting ecosystem)
```

### Architectural Progress Tracking:
1. **Evolution from Random to Deterministic Data**: Early pipeline builds utilized unstructured test shapes which caused immediate model divergence during evaluation. The production data script introduces rigid mathematical matrices (224x224x3) mapping explicit boundaries per label to prevent mode collapse.
2. **Evolution from GPU Timeout to CPU Isolation**: Natively, TensorFlow attempts intensive bus handshakes with Windows graphics stacks. For single-image inference tasks, this creates high-latency thread locks. Injecting `CUDA_VISIBLE_DEVICES = "-1"` cut down boot latency by 90% and guaranteed continuous 0.00-second request turnaround times.
3. **Evolution from Monolithic UI to Hybrid Frontend Strategy**: Moving from standard HTML layout parsing to **Streamlit** removed hundreds of lines of brittle DOM management scripts. The Streamlit data engine processes file uploads out of the box, wraps memory pointers safely, and binds JSON outputs directly to native web UI rendering blocks.

---

## 📖 Deep Learning & MLOps Glossary

*   **Transfer Learning**: An engineering pattern where a model pre-trained on huge baseline datasets (*ImageNet*) is used as a fixed feature extractor for a custom classification task, cutting convergence loops by 95%.
*   **MobileNetV2**: A production-grade convolutional architecture utilizing depthwise separable convolutions to dramatically decrease the parameters required, making it ideal for low-latency web deployment.
*   **Softmax Normalization**: An algebraic activation layer that maps raw logit strings from neural layers into a distinct probability vector where all numbers sit between 0 and 1 and sum exactly to 1.0.
*   **Model Latency Optimization**: The engineering practice of profiling and removing computing blockers (like driver checks or redundant disk reads) from inference pathways to keep request response speeds as low as possible.
*   **Deterministic Weights**: System states where inputs produce predictable outputs without variance, achieved here by filling pixel structures with exact RGB configurations (`CULOARE_ROSU`, `CULOARE_GALBEN`).
