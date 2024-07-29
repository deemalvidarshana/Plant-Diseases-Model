# Plant Diseases Model

This repository contains a Jupyter notebook for building and evaluating a deep learning model to detect plant diseases.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

The `PLANT_DISEASES__MODEL2.ipynb` notebook provides an end-to-end pipeline for preprocessing data, training a deep learning model, and evaluating its performance in detecting plant diseases.

## Installation

To run this notebook, you'll need:

- Python 3.7+
- Jupyter Notebook
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow/keras

Install the required packages using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/deemalvidarshana/plant-diseases-model.git
```

2. Navigate to the project directory:

```bash
cd plant-diseases-model
```

3. Open the Jupyter notebook:

```bash
jupyter notebook PLANT_DISEASES__MODEL2.ipynb
```

4. Follow the steps in the notebook to preprocess the data, train the model, and evaluate its performance.

## Dataset

The dataset can be downloaded from [Kaggle](https://www.kaggle.com/emmarex/plantdisease). It contains images of healthy and diseased plant leaves categorized into different classes.

## Model

The model is a Convolutional Neural Network (CNN) built with TensorFlow/Keras. It includes convolutional layers, max-pooling layers, and fully connected layers, trained using the Adam optimizer and categorical cross-entropy loss.

## Results

The trained model achieved an accuracy of 95% on the validation set. Detailed performance metrics and visualizations are available in the notebook.

## Contributing

Fork the repository and submit a pull request for improvements, bug fixes, or new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

Replace `https://github.com/deemalvidarshana/plant-diseases-model.git` 