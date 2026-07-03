# 📘 Project Architecture Data Flow & Technical Glossary

This document serves as an exhaustive operational manual detailing the pipeline execution flow and architectural terminology utilized throughout the AgriVision AI application.

---

## 🔄 End-to-End Project Data Flow

The engineering pipeline operates across four distinct structural layers to process data from generation to live web production:

```text
[ 1. DATA PATH ]     Synthetic Image Synthesis (scripts/download_real_images.py)
                            │  (Outputs stratified geometric shapes to dataset_fructe/)
                            ▼
[ 2. TRAIN PATH ]    Deep Learning Pipeline (scripts/train_pipeline.py)
                            │  (Feeds images via ImageDataGenerator -> MobileNetV2 Base)
                            │  (Serializes optimal weights to models/fruct_model_complet.keras)
                            ▼
[ 3. BACKEND PATH ]  Flask REST API (app.py)
                            │  (Loads .keras model on CPU context to avoid GPU timeouts)
                            │  (Exposes POST /predict endpoint waiting for web payloads)
                            ▼
[ 4. FRONTEND PATH ] HTML5 / Vanilla JS Client Dashboard (frontend/index.html)
                               (Processes file selection input -> Fetch API -> Animates Probability Bars)
```

1. **Data Synthesis Layer (`scripts/download_real_images.py`)**: To solve cold-start data scarcity, this module programmatically builds an expandable dataset (40 train / 10 val images per class) embedding geometric abstractions (circles/polygons) mapped to fixed RGB tensors.
2. **Model Compilation & Training Layer (`scripts/train_pipeline.py`)**: Tensors are streamed via `ImageDataGenerator` and passed through a frozen **MobileNetV2** feature extractor. Custom classification heads compute feature distributions and serialize the graph to the `models/` directory.
3. **Inference Serving Layer (`app.py`)**: A production-grade Flask instance abstracts the Keras model execution. It reads binary image streams from incoming HTTP requests, standardizes the pixel resolution to `(224, 224, 3)`, maps inputs through NumPy vector axes, and responds with structured JSON.
4. **User Experience Layer (`frontend/index.html`)**: The client application handles multi-channel asset streaming via manual file explorer picking, queries the backend asynchronously via the **Fetch API**, and parses probabilities directly into UI elements.

---

## 📖 Deep Learning & MLOps Glossary

*   **Convolutional Neural Network (CNN)**: A specialized deep learning architecture utilizing weight-sharing convolutional kernels to preserve spatial hierarchies and extract local pixel patterns (edges, textures) from visual datasets.
*   **Transfer Learning**: A paradigm where a model checkpoint pre-trained on a massive dataset (e.g., *ImageNet*) is repurposed as a foundational feature extractor for a new, domain-specific task, drastically reducing convergence time.
*   **MobileNetV2**: A highly efficient convolutional neural network architecture optimized for mobile and edge deployment, relying on depthwise separable convolutions to minimize computational parameters.
*   **Categorical Crossentropy**: The mathematical loss function used in multi-class classification problems to evaluate the divergence between the true one-hot encoded label distribution and the model's soft probability outputs.
*   **Softmax Activation**: An algebraic function applied to the final output layer of a multi-class network that normalizes raw logits into a probabilistic distribution summing exactly to 1.0.
*   **ImageDataGenerator**: A Keras utility used to orchestrate real-time data streaming, rescaling pixel values to standardized ranges (e.g., `1.0 / 255`) directly during tensor training loops.
*   **Lazy Loading (Model Context)**: An MLOps optimization pattern where heavy binary model weights are not compiled into system memory during initial application booting, but compiled *lazily* upon the arrival of the first API request.
*   **Model Re-tracing Warning**: A TensorFlow performance log indicating that a compiled graphical function (`@tf.function`) is repeatedly regenerating execution paths, usually caused by passing varying NumPy structural shapes into the execution graph.
*   **Overfitting**: An optimization failure where a model learns training noise and irrelevant dataset variances instead of underlying features, causing training accuracy to skyrocket while validation metrics plummet.
*   **Early Stopping**: A regularization callback that monitors verification loss loops and terminates the model training sequence early if the network stops improving, saving computing overhead and preventing over-optimization.
*   **TensorBoard**: A visualization framework utilized to map, profile, and track quantitative performance scalar metrics (Loss and Accuracy curves) across successive deep learning iterations.
