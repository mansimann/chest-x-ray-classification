# :lungs: Chest-X-Ray Classification 

This project classifies chest X-ray images using a modified ResNet-50 neural network model.

## Contents
 
- [Technologies](#technologies)
- [Dataset description](#dataset-description)
- [Data preprocessing](#data-preprocessing) 
- [Proposed model](#proposed-model)
- [Methodology](#methodology) 
- [Results](#results) 
- [Code style](#code-style)
- [Requirements](#requirements)
- [Credits](#credits)

## Technologies

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)

## Dataset description

The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database). 
Researchers from Qatar University and University of Dhaka, along with their collaborators from Pakistan and Malaysia, 
created this dataset of chest X-ray (CXR) images. 

They released 3616 COVID, 6012 Lung Opacity, 10192 Normal, and 1345 Viral Pneumonia CXRs, totaling 21265.

<img height="400" src="figures/bar_chart.png" width="400"/><img height="400" src="figures/pie_chart.png" width="400"/>

Every image was PNG, grayscale, and 299 x 299 pixels. 

<img height="200" src="figures/dataset_images.png" width="800"/>

## Data preprocessing

Images were processed using `tf.keras.applications.resnet50.preprocess_input`, which converts images from RGB to BGR 
and then zero-centers each color channel with respect to the ImageNet dataset without scaling.

In addition, real-time data augmentation was done. The table below details the parameters used.

<img height="180.5" src="figures/aug_parameters_summary.png" width="509"/>

For visualization purposes, some augmented pictures were saved to the aug_images folder via the `save_to_dir` arg.

<img height="160" src="aug_images\aug_809_465415.png" width="160"/><img height="160" src="aug_images\aug_6868_6483323.png" width="160"/><img height="160" src="aug_images\aug_7903_3196687.png" width="160"/><img height="160" src="aug_images\aug_9366_9241711.png" width="160"/><img height="160" src="aug_images\aug_4010_3591919.png" width="160"/>

## Proposed model

First, a ResNet-50 neural network model was implemented from scratch and 
pre-trained ImageNet weights were loaded into it via transfer learning. 
Next, bottleneck layers were added. The code block details the modifications. 

```python
x = base_model.output
x = Conv2D(1024, 1)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
outputs = Dense(4, activation='softmax')(x)
model = Model(base_model.input, outputs)
```

Then, the base model layers were frozen,
rendering 2,755,332 trainable parameters and 23,587,712 non-trainable parameters.
Below the red line are all the modifications.

<img height="384.5.5" src="figures/model_summary.png" width="632.5"/>

## Methodology

The table below summarizes the dataset partition details.

<img height="153" src="figures/dataset_partition.png" width="590"/>

First, class weights were estimated for the unbalanced dataset.
The model was then was fed batches of augmented images. 
The images were in rgb mode and 224 x 224 in size for ImageNet compatibility reasons.
These images were trained for 25 epochs with a  learning rate of 0.0001. 
The learning rate was lowered to 0.00001 for the last 5 epochs.

## Results

### Training Results

<img height="300" src="figures/Loss.png" width="400"/> <img height="300" src="figures/Accuracy.png" width="400"/>
<img height="300" src="figures/Precision.png" width="400"/> <img height="300" src="figures/Recall.png" width="400"/>

### Testing Results 

After training, the test subset was evaluated. 
The table below summarizes the metrics for the model in test mode.

<img height="52.5" src="figures/test_results.png" width="473"/>

Next, output predictions were generated for the test input samples. 
These predictions were then compared to the true label values.
The classification report and confusion matrix below summarize the results. 

See `sklearn.metrics.classification_report` and `sklearn.metrics.ConfusionMatrixDisplay` for more details. 

<img height="222.5" src="figures/classification_report.png" width="471.5"/>

<img height="533.3" src="figures/confusion_matrix.png" width="800"/>

## Code style

![Custom badge](https://img.shields.io/badge/code%20style-PEP%208-brightgreen)
![Custom badge](https://img.shields.io/badge/docstring%20format-reStructuredText-brightgreen)

## Requirements

Third party / library specific imports: matplotlib, numpy, IPython, PIL, sklearn, tensorflow

## Credits

- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition (Version 1). arXiv. https://doi.org/10.48550/ARXIV.1512.03385
- M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676. 
- Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S.B.A., Islam, M.T., Maadeed, S.A., Zughaier, S.M., Khan, M.S. and Chowdhury, M.E., 2020. Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images.