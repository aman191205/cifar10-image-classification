# Deep Learning Assignment 1 — CIFAR-10 Image Classification

## 1. Project Overview

This project implements an image classification system using TensorFlow/Keras and the CIFAR-10 dataset. The main objective is to study how data augmentation and regularization techniques affect classification accuracy and model robustness against image corruption.

The notebook uses a subset of 500 images for training and validation, compares different augmentation strategies, evaluates models on clean and shifted test images, and performs error analysis and test-time augmentation.

## 2. Dataset

The notebook automatically downloads the CIFAR-10 dataset using `tf.keras.datasets.cifar10.load_data()`.

CIFAR-10 contains 32 × 32 RGB images belonging to 10 classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

A subset of 500 images is selected, with 50 images from each class.

| Dataset | Number of Images |
|---|---:|
| Training set | 400 |
| Validation set | 100 |
| Clean test set | Original CIFAR-10 test set |
| Shifted test set | Corrupted test images |

The training and validation sets preserve the class distribution. The original test set is kept separate from training and validation.

## 3. Implementation

### 3.1 Environment and Reproducibility

The notebook uses TensorFlow, NumPy, Pandas, Matplotlib, and Scikit-learn. Fixed random seeds are used to make the experiments reproducible.

The experiments use three seeds: `42`, `123`, and `2026`. Each model configuration is evaluated across these seeds to compare performance and variation between runs.

### 3.2 Shifted Test Dataset

A shifted-test dataset is generated from the original test images using image corruptions such as:

- Random rotation
- Brightness and contrast changes
- Gaussian noise
- Black rectangular occlusion

This dataset is used to evaluate model robustness against image corruption. It is a locally generated approximation rather than an official corruption benchmark.

### 3.3 Data Augmentation

Different augmentation configurations are compared:

1. **No Augmentation:** Uses the original training images.
2. **Geometric Augmentation:** Applies random crops, horizontal flips, rotations, and translations.
3. **Photometric Augmentation:** Applies brightness and contrast changes, blur, and noise.
4. **Combined Augmentation:** Combines geometric and photometric transformations.
5. **Cutout / Random Erasing:** Masks random image regions to simulate partial occlusion.
6. **Mixup and Label Smoothing:** Mixes training examples and their labels to improve generalization.
7. **Final Model:** Uses the selected augmentation and regularization techniques.

Augmentation is applied during training and is not applied to the validation and test datasets.

### 3.4 Model Architecture

A compact Inception-style convolutional neural network is implemented using TensorFlow/Keras.

The model uses parallel convolutional branches with different kernel sizes to extract image features at multiple scales. Batch Normalization, Dropout, and AdamW with weight decay are used to improve training stability and reduce overfitting.

### 3.5 Training Strategy

The notebook uses the following training settings:

- Batch size: 32
- Maximum training epochs: 20
- Experiment seeds: 42, 123, and 2026
- Optimizer: AdamW
- Early Stopping patience: 5 epochs
- Learning rate scheduling: ReduceLROnPlateau
- Mixup label smoothing: 0.05

Early Stopping monitors validation accuracy and restores the best weights. ReduceLROnPlateau reduces the learning rate when validation loss stops improving.

### 3.6 Evaluation and Analysis

Each model configuration is evaluated on clean and shifted test images. The notebook records accuracy, execution time, and other evaluation metrics.

The analysis includes:

- Training and validation accuracy curves
- Training and validation loss curves
- Clean-test and shifted-test accuracy comparisons
- Mean and standard deviation across experiment seeds
- Per-class accuracy on shifted images
- A normalized confusion matrix
- Misclassified image examples
- Grayscale evaluation as a proxy for hidden corruption
- Test-time augmentation using original and horizontally flipped images

The final configuration is selected using validation performance rather than test-set results.

## 4. Requirements

The project requires the following Python libraries:

- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

Google Colab with a GPU runtime is recommended because multiple model configurations are trained across several seeds.

## 5. Instructions to Reproduce the Results

### Step 1: Open the Notebook

Open `DL_ASS01___23F_0738_.ipynb` in Google Colab or a local Jupyter Notebook environment.

### Step 2: Configure the Runtime

For Google Colab:

1. Open the notebook.
2. Select **Runtime → Change runtime type**.
3. Select a GPU runtime, such as T4 GPU, if available.
4. Run the following code in a cell to verify GPU availability:

```python
import tensorflow as tf

print(tf.config.list_physical_devices("GPU"))
```

### Step 3: Install Dependencies

Google Colab normally includes the required libraries. For a local environment, install them using:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn jupyter
```

### Step 4: Run the Notebook

Run all notebook cells in their original order.

The notebook will:

1. Import the required libraries and configure random seeds.
2. Download the CIFAR-10 dataset automatically.
3. Select the 500-image subset.
4. Create training and validation sets.
5. Generate the shifted-test dataset.
6. Define the CNN architecture and augmentation techniques.
7. Train the baseline and augmented models.
8. Evaluate the models on clean and shifted test images.
9. Generate comparison tables, graphs, and error-analysis results.
10. Evaluate grayscale images and test-time augmentation.
11. Execute the final checks included in the notebook.

No external dataset files are required because CIFAR-10 is downloaded automatically.

### Step 5: Review the Results

After all cells finish executing, review the generated tables and plots.

- Compare the baseline with the augmented models.
- Examine validation accuracy across the different configurations.
- Compare clean-test and shifted-test accuracy to assess robustness.
- Review the confusion matrix and misclassified images.
- Compare normal inference with test-time augmentation.
- Examine the mean and standard deviation across the three experiment seeds.

The final numerical results are generated by executing the notebook and are not fixed in this README.

## 6. Reproducibility Notes

- Use the same experiment seeds: `42`, `123`, and `2026`.
- Keep the dataset selection and train-validation split unchanged.
- Run the notebook from the beginning to initialize all datasets and models correctly.
- Keep the augmentation and corruption-generation code unchanged for consistent comparisons.
- GPU availability, TensorFlow versions, and hardware differences may affect execution time and cause small numerical variations.
- The grayscale evaluation is a proxy test and should not be presented as an official hidden-corruption benchmark.
- The notebook includes checks for the 500-image subset, the 400/100 training-validation split, the model parameter limit of one million, and the epoch limit.

## 7. Expected Results

After successful execution, the notebook produces:

- Trained CNN models
- Model comparison tables
- Training and validation learning curves
- Clean-test and shifted-test accuracy measurements
- Per-class accuracy results
- A normalized confusion matrix
- Misclassified image examples
- Grayscale evaluation results
- Test-time augmentation performance

These outputs help evaluate how data augmentation and regularization affect classification performance and robustness to image corruption.

## 8. Conclusion

This project demonstrates an end-to-end deep learning workflow for CIFAR-10 image classification. By comparing different augmentation and regularization techniques under consistent experimental conditions, it investigates their effects on model generalization and robustness against image corruption.
