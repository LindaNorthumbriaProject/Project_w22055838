
# Linda Trials

This repository contains Jupyter notebooks and datasets related to the implementation and evaluation of Convolutional Neural Networks (CNNs) for facial expression recognition using various datasets. The notebooks explore different model architectures, data augmentation techniques, and training configurations to optimize the performance of the models on the respective datasets.

## Folder Structure

- **Datasets**:
  - **BU_3DFE_Database_sorted/**: Directory containing the BU_3DFE dataset, sorted by emotion labels.
  - **JAFFE Dataset/**: Directory containing the JAFFE dataset images for facial expression recognition.
  - **Bosphorus Dataset/**: Directory containing the Bosphorus dataset images for facial expression recognition.
  - **CK+ Dataset/**: Directory containing the CK+ dataset images for facial expression recognition.

- **Jupyter Notebooks**:
  - **bu_final.ipynb**: Implementation and evaluation of CNN models using the BU_3DFE dataset.
  - **jafee-final.ipynb**: Implementation and evaluation of CNN models using the JAFFE dataset.
  - **bosphorus_final.ipynb**: Implementation and evaluation of CNN models using the Bosphorus dataset.
  - **ck+-final.ipynb**: Implementation and evaluation of CNN models using the CK+ dataset.

## Notebooks Overview

### 1. `bu_final.ipynb`
- **Focus**: Implementation of CNN architectures with detailed parameters and training configurations using the BU_3DFE dataset.
- **Key Sections**:
  - **Dataset Preparation**: Loading, preprocessing, and splitting the dataset into training and testing sets.
  - **Class Weights**: Handling class imbalance by computing class weights.
  - **Data Augmentation**: Applying height shift augmentation.
  - **CNN Architecture**: A model with 2 convolutional layers (32 and 64 filters), max-pooling, and a fully connected layer with 100 neurons.
  - **Training and Evaluation**: Training the model with data augmentation, calculating precision, recall, F1 score, and generating a confusion matrix.
  - **Visualization**: Plotting training accuracy, loss, and confusion matrix.

### 2. `jafee-final.ipynb`
- **Focus**: Dataset preprocessing, CNN architecture, and model training with parameter exploration on the JAFFE dataset.
- **Key Sections**:
  - **Dataset Preprocessing**: Loading, preprocessing, and splitting the dataset into training and testing sets.
  - **CNN Architecture**: A base model with 2 convolutional layers, max-pooling, and a fully connected layer, with variations in parameters across different configurations.
  - **Training Configurations**: Exploring different configurations by varying the number of filters, dense units, dropout rates, and training parameters.
  - **Data Augmentation**: Applying techniques such as horizontal flips, zooms, and rotations.
  - **Evaluation and Visualization**: Calculating accuracy, precision, recall, F1 score, plotting accuracy/loss, and generating confusion matrices.

### 3. `bosphorus_final.ipynb`
- **Focus**: CNN model training with data augmentation and evaluation using the Bosphorus dataset.
- **Key Sections**:
  - **Dataset Preparation**: Loading, preprocessing, and splitting the dataset into training and testing sets.
  - **Class Weights**: Handling class imbalance by computing class weights.
  - **CNN Architecture**: A model with 2 convolutional layers, max-pooling, and a fully connected layer with 100 neurons.
  - **Training and Evaluation**: Training the model with data augmentation and evaluating its performance.
  - **Visualization**: Displaying accuracy, loss, and confusion matrix.

### 4. `ck+-final.ipynb`
- **Focus**: Implementation and evaluation of various CNN architectures on the CK+ dataset.
- **Key Sections**:
  - **Dataset Preprocessing**: Loading, preprocessing, and splitting the dataset into training and testing sets.
  - **Class Weights**: Handling class imbalance by computing class weights.
  - **CNN Architecture**: Multiple configurations with varying convolutional layers, dense units, and dropout rates.
  - **Data Augmentation**: Applying techniques like height shifts and minimal rotations.
  - **Training and Evaluation**: Training the model with early stopping, evaluating performance, and visualizing results.
  - **Visualization**: Plotting accuracy, loss, and confusion matrices for different configurations.

## How to Run

1. **Install Dependencies**:
   Ensure you have Python 3.x installed. Install the required dependencies using pip:

   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset Preparation**:
   Ensure that the datasets (e.g., BU_3DFE, JAFFE, Bosphorus, CK+) are placed in their respective folders as mentioned in the folder structure.

3. **Running Notebooks**:
   - Open the notebooks using Jupyter Notebook or JupyterLab.
   - Run each cell in the notebooks to execute the code and observe the results.

4. **Results**:
   The notebooks will output evaluation metrics, accuracy/loss plots, and confusion matrices. The visualizations will be displayed inline within the notebooks.

## Requirements

- Python 3.x
- Jupyter Notebook / JupyterLab
- TensorFlow
- NumPy
- PIL (Pillow)
- scikit-learn
- matplotlib
- seaborn

## Results and Analysis

Each notebook generates detailed outputs, including:
- **Accuracy and Loss Plots**: Showing the training and validation performance across epochs.
- **Confusion Matrices**: Visualizing the model's performance across different classes.
- **Metric Summaries**: Including test accuracy, precision, recall, and F1 scores for each configuration.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
