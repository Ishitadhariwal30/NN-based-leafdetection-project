

#  U-Net Autoencoder for Leaf Image Reconstruction

This project implements a **deep learning-based U-Net autoencoder** to reconstruct grayscale images of leaves collected from the MBMU campus. It uses a convolutional architecture optimized for unsupervised image reconstruction tasks, particularly useful for denoising, anomaly detection, or generating clean versions of biological image data.

---

##  Dataset

* **Input Data:** Preprocessed grayscale images of size `256x256` pixels stored in a `.npy` file.
* **File used:** `/content/drive/MyDrive/preprocessed_images.npy`
* **Total Images:** 5154 grayscale leaf images

---

##  Model Architecture

The model is a **U-Net-style autoencoder** built using TensorFlow/Keras, with the following structure:

###  Encoder

* Conv2D → Dropout → Conv2D → MaxPooling (×3 blocks)
* Filters: 16 → 32 → 64
* Regularization: L2 (`1e-4`) on all convolutional layers

###  Bottleneck

* Two Conv2D layers with 128 filters
* Dropout rate: 0.3

###  Decoder

* UpSampling2D → Concatenate with encoder output → Conv2D (×3 blocks)
* Filters: 64 → 32 → 16
* Final layer: `Conv2D` with `sigmoid` activation to output reconstructed image

---

##  Features

*  **Input Shape:** `(256, 256, 1)`
*  **Loss Function:** Binary Crossentropy
*  **Optimizer:** Adam (`lr=1e-4`)
*  **EarlyStopping:** Stops training if validation loss doesn’t improve after 5 epochs
*  **Metrics:** Mean Squared Error (MSE)
*  **Training/Validation Split:** 70% training, 30% testing
*  **Data Augmentation (Optional):** Rotation, shift, zoom, flip
*  **Total Parameters:** 441,816
*  **Visualization:** Loss curves and reconstruction error histogram
*  **Evaluation:** MSE, RMSE, and reconstruction error distribution on test data

---

##  Training Details

```python
autoencoder.fit(
    x_train, x_train,
    epochs=15,
    batch_size=16,
    shuffle=True,
    validation_data=(x_test, x_test),
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)
```

---

##  Results

* **Test MSE:**  0.0000
* **Test RMSE:** 0.0043
* **Visualization:**

  *  Training vs Validation Loss
  *  Histogram of reconstruction error (per image MSE)

---

##  Sample Visualization

* Original and reconstructed images can be displayed using:

```python
plt.imshow(decoded_imgs[i].reshape(256, 256), cmap='gray')
```

---

##  How to Run

1. Upload your `preprocessed_images.npy` to Google Drive.
2. Mount Google Drive in Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Load and normalize the data, reshape to `(samples, 256, 256, 1)`.
4. Define the model using `unet_autoencoder()`.
5. Train and evaluate.

---

##  Dependencies

* Python 3.x
* NumPy
* Matplotlib
* Scikit-learn
* TensorFlow (>=2.x)

Install them via:

```bash
pip install numpy matplotlib scikit-learn tensorflow
```

---

##  Future Work

* Include reconstruction comparison plots
* Integrate anomaly detection by thresholding reconstruction error
* Use Variational Autoencoder (VAE) extension for generative analysis
* Experiment with other regularization techniques like `dropout`, `batch normalization`



