# 💼 Technical Interview Preparation Folder (AI/ML Developer)

This document contains targeted, advanced technical questions and professional-grade answers mapping directly to the architectural decisions made in the AgriVision AI project.

---

### Q1: Why did you choose Transfer Learning with MobileNetV2 instead of training a custom CNN from scratch?
**Ideal Answer:**
"I selected **MobileNetV2** via Transfer Learning because it strikes an optimal balance between feature extraction depth and computational efficiency. Training a custom CNN from scratch on a small dataset often leads to rapid overfitting or slow convergence due to unoptimized initial weights. By freezing the MobileNetV2 base pre-trained on ImageNet, the model leverages highly sophisticated, low-level feature detectors (edges, gradients, textures) right away. This allowed me to achieve a 100% validation accuracy within just a few epochs on the custom classification head, minimizing training time and compute resource overhead."

### Q2: In your backend `app.py`, why did you explicitly set the environment variable `CUDA_VISIBLE_DEVICES = "-1"`?
**Ideal Answer:**
"In Windows and hybrid production environments, TensorFlow natively attempts to scan system hardware buses for compatible NVIDIA CUDA cores during execution booting. If a dedicated GPU isn't available, or if driver versions mismatch, this hardware probing triggers long thread timeouts, causing the Flask web server to hang indefinitely at launch. By explicitly forcing `CUDA_VISIBLE_DEVICES = "-1"`, I forced TensorFlow to bypass GPU initialization entirely and run directly on the host CPU. For low-latency inference tasks with small batch sizes ($N=1$), CPU execution path profiling eliminates driver overhead and ensures robust, instantaneous API responses."

### Q3: Your Flask backend uses `get_production_model()` with a global variable check. What is this design pattern called and what problem does it solve?
**Ideal Answer:**
"This is an implementation of the **Lazy Loading (or Singleton initialization)** pattern. Compiling a Keras `.keras` binary file into memory requires intensive disk I/O operations and graphical execution mapping. If the model loading logic were placed directly inside the global scope of `app.py` or repeated within the `/predict` route, it would block the Flask application boot sequence or severely degrade performance on every incoming API request. By checking if `model_instance is None`, the application boots instantly. The model is compiled into RAM exactly once during the very first request and remains warm in cache memory to serve all subsequent user queries with sub-millisecond overhead."

### Q4: Explain the difference between `predicted_class` and `distribution` in your JSON payload, and why returning the full distribution is an MLOps best practice.
**Ideal Answer:**
"The `predicted_class` represents the deterministic output—the single class with the highest probability index calculated via `np.argmax()`. The `distribution` dictionary exposes the raw outputs of the network's final **Softmax** layer for every monitored class. Returning the full distribution is a critical production best practice because it enables **downstream confidence auditing**. For instance, if a model predicts 'Apple' with 35% confidence, but 'Orange' holds 33%, a system relying only on the top text output might pass a faulty inference. By serving the complete probability array, the client app or a threshold monitoring system can intercept low-margin predictions and flag them for manual review or data drift analysis."

### Q5: During package management, why did you opt for `uv` over standard `pip` + `venv`, and how did you resolve Python version mismatches?
**Ideal Answer:**
"I implemented Astral's **`uv`** ecosystem because it resolves virtual environment packages up to 10-100 times faster than traditional Python packaging tools. A major technical roadblock in this project was that TensorFlow 2.16.1 does not provide binary distributions or stable wheels for Python 3.14. Using traditional tools would require manual Python compilation. With `uv`, I used the command `uv venv --python 3.12`, which dynamically sources, downloads, and sandboxes an entirely isolated Python 3.12 binary directly inside the project root workspace, completely bypassing global host machine version constraints cleanly and safely."

### Q6: How does your frontend JavaScript manage file state, and why did you bypass using standard web hosting protocols like HTTP for the client file?
**Ideal Answer:**
"The frontend leverages HTML5 **FileReader API** and event listener captures on the drop zone to abstract the physical file reference directly from browser contexts into memory as a DataURL asset for instant layout previewing. When sending data, it embeds the raw file binary into a native `FormData` key-value multi-part payload matching the multi-part structure expected by the Flask REST engine. I intentionally kept the frontend running on the local file system path context (`C:/.../index.html`) rather than an isolated web proxy layer to isolate client code execution and ensure zero-configuration deployments during development showcase pipelines."

