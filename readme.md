# Deep Crowns: Interpretability of Wildfire Spread Models

## Overview

This project focuses on applying interpretability techniques, such as Grad-CAM, to a U-Net based model trained on wildfire spread data. The goal is to understand the conditions under which extreme wildfire events occur.

## Introduction

Wildfires are a significant environmental hazard, and understanding the conditions that lead to extreme wildfire events is crucial for prevention and mitigation. This project leverages a U-Net based model to predict wildfire spread and employs interpretability techniques to gain insights into the model's decision-making process.

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/deep-crowns.git
cd deep-crowns
pip install -r requirements.txt
```

## U-Net

The U-Net model is a convolutional neural network architecture that is widely used for image segmentation tasks. It consists of an encoder-decoder structure with skip connections that help preserve spatial information during the upsampling process.


### Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to produce visual explanations for the predictions made by the U-Net model. It highlights the regions in the input data that are most influential in the model's decision-making process.


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.