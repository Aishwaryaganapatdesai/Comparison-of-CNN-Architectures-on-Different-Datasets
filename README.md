# Comparison-of-CNN-Architectures-on-Different-Datasets
This project compares the performance of seven CNN architectures (LeNet-5, AlexNet, GoogLeNet, VGGNet, ResNet, Xception, and SENet) on three image classification datasets (MNIST, FMNIST, CIFAR-10). It evaluates models based on accuracy, precision, recall, F1-score, and loss curves, helping to choose the best model for specific tasks.

# **Comparison of CNN Architectures on Different Datasets**

## **Project Overview**
This project compares the performance of various Convolutional Neural Network (CNN) architectures on three popular image classification datasets: MNIST, FMNIST, and CIFAR-10. The goal is to evaluate seven CNN architectures: LeNet-5, AlexNet, GoogLeNet, VGGNet, ResNet, Xception, and SENet on these datasets. The performance comparison will be based on metrics such as accuracy, precision, recall, F1-score, and loss curves.

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Datasets](#datasets)
3. [Model Architectures](#model-architectures)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Training](#training)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Results](#results)
9. [Project Deliverables](#project-deliverables)
10. [Contributing](#contributing)
11. [License](#license)
12. [Project Structure](#project-structure)

## **Datasets**
The following datasets are used in this project:
- **MNIST**: A dataset of 28x28 grayscale images of handwritten digits (0-9). It contains 60,000 training images and 10,000 test images.
- **FMNIST**: A dataset similar to MNIST, but with images of clothing items. It consists of 60,000 training images and 10,000 test images.
- **CIFAR-10**: A more complex dataset consisting of 60,000 32x32 color images in 10 classes (e.g., airplanes, cars, dogs). It includes 50,000 training images and 10,000 test images.

## **Model Architectures**
This project evaluates the following CNN architectures:
1. **LeNet-5**: A classic CNN architecture, simple but effective for small images like MNIST.
2. **AlexNet**: A deeper architecture that brought CNNs into the spotlight, used for larger and more complex datasets like CIFAR-10.
3. **GoogLeNet**: Uses inception modules to allow the network to learn features at multiple scales.
4. **VGGNet**: Known for its simplicity and depth, utilizing small 3x3 convolutions stacked on top of each other.
5. **ResNet**: A deep architecture that uses residual connections to combat the vanishing gradient problem.
6. **Xception**: An extension of Inception that uses depthwise separable convolutions.
7. **SENet**: Incorporates attention mechanisms to adaptively recalibrate channel-wise feature responses.

## **Installation**
To run this project, you need to have Python 3.6+ installed along with PyTorch and TensorFlow. The following steps outline the installation process:

### 1. Clone the repository:
```bash
git clone https://github.com/yourusername/cnn-architectures-comparison.git
cd cnn-architectures-comparison
```

### 2. Install dependencies:
Use `pip` to install the required libraries.
```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains the following:
- torch
- torchvision
- tensorflow
- matplotlib
- numpy
- scikit-learn
- pandas

## **Usage**
### 1. Preprocessing the datasets:
The datasets (MNIST, FMNIST, CIFAR-10) will be automatically downloaded during the first run. Ensure that the images are resized and normalized to match the model's input requirements.

### 2. Running the models:
You can choose to train and evaluate any of the seven CNN models. To run the script, use the following command:

```bash
python train.py --model <model_name> --dataset <dataset_name>
```

Replace `<model_name>` with one of the following:
- lenet5
- alexnet
- googlenet
- vggnet
- resnet
- xception
- senet

Replace `<dataset_name>` with one of the following:
- mnist
- fmnist
- cifar10

### Example command:
```bash
python train.py --model resnet --dataset cifar10
```

This command will train the ResNet model on the CIFAR-10 dataset.

## **Training**
### Model Training:
Each model is trained separately on each dataset. The training process involves the following steps:
1. Load and preprocess the dataset.
2. Define the CNN architecture.
3. Train the model using the training dataset.
4. Save the trained model weights.
5. Plot loss curves and evaluate performance metrics.

### Hyperparameters:
The training script allows you to specify hyperparameters such as:
- Learning rate
- Batch size
- Number of epochs
- Optimizer (e.g., Adam, SGD)

You can adjust these parameters in the `train.py` file or use command-line arguments.

## **Evaluation Metrics**
The performance of the models is evaluated using the following metrics:
- **Accuracy**: The percentage of correct predictions made by the model.
- **Precision**: The proportion of true positive predictions out of all positive predictions.
- **Recall**: The proportion of true positive predictions out of all actual positives.
- **F1-score**: The harmonic mean of precision and recall, providing a balance between both metrics.
- **Loss**: The value of the loss function during training, indicating how well the model is performing.

Metrics will be plotted after training for visual comparison of model performance.

## **Results**
The expected outcomes from the project include:
- Loss curves for each model on each dataset.
- Evaluation metrics (accuracy, precision, recall, F1-score) for each model.
- A comparative analysis of the models' performance on MNIST, FMNIST, and CIFAR-10 datasets.

## **Project Deliverables**
Upon completion of the project, the following should be submitted:
- **Source Code**: Python scripts for implementing and training the models.
- **Documentation**: Detailed approach, results, and analysis.
- **Plots**: Loss curves and performance metrics visualizations.
- **Final Report**: Summary of findings and conclusions.

## **Contributing**
If you'd like to contribute to this project, please fork the repository, make changes, and submit a pull request with a description of your changes. All contributions are welcome!

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## **Project Structure**
Below is the directory structure for the project:

```
CNN_project/
├── data/                     # Directory for datasets
│   ├── MNIST/                # MNIST dataset
│   │   ├── raw/              # Raw dataset files
│   │   │   ├── train-images.idx3-ubyte
│   │   │   ├── train-labels.idx1-ubyte
│   │   │   ├── t10k-images.idx3-ubyte
│   │   │   ├── t10k-labels.idx1-ubyte
│   ├── FMNIST/               # FMNIST dataset
│   │   ├── raw/              # Raw dataset files
│   │   │   ├── train-images-idx3-ubyte
│   │   │   ├── train-labels-idx1-ubyte
│   │   │   ├── t10k-images-idx3-ubyte
│   │   │   ├── t10k-labels-idx1-ubyte
│   ├── CIFAR-10/             # CIFAR-10 dataset
│   │   ├── raw/              # Raw dataset files
│   │   │   ├── cifar-10-batches-bin/
│   │   │   ├── cifar-10-python.tar.gz
├── models/                   # Directory for CNN model architectures
│   ├── lenet.py              # LeNet-5 implementation
│   ├── alexnet.py            # AlexNet implementation
│   ├── googlenet.py          # GoogLeNet implementation
│   ├── vggnet.py             # VGGNet implementation
│   ├── resnet.py             # ResNet implementation
│   ├── exception.py          # Xception implementation
│   ├── senet.py              # SENet implementation
├── tests/                    # Directory for model testing scripts
│   ├── test_lenet.py         # Test script for LeNet-5
│   ├── test_alexnet.py       # Test script for AlexNet
│   ├── test_googlenet.py     # Test script for GoogLeNet
│   ├── test_vggnet.py        # Test script for VGGNet
│   ├── test_resnet.py        # Test script for ResNet
│   ├── test_exception.py     # Test script for Xception
│   ├── test_senet.py         # Test script for SENet
│   └── run_tests.py          # Script to run all tests together
├── notebooks/                # (Optional) For Jupyter Notebooks
├── plots/                    # Save plots and graphs here
├── utils/                    # Directory for utility functions
│   ├── data_loader.py        # Dataset loading and preprocessing
│   ├── metrics.py            # Functions for evaluation metrics
│   ├── visualization.py      # Functions for plotting results
├── main.py                   # Main script to run the project
├── requirements.txt          # Dependencies for the project
├── README.md                 # Project documentation
├── LICENSE                   # (Optional) License file
└── .gitignore                # Git ignore file

```

