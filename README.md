# HSST Workshop: Auto-segmentation

## Part 1:
<a target="_blank" href="https://colab.research.google.com/github/Christie-Scientific-Computing/HSST_Workshop/blob/main/Part_1_abdominal_autoseg.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Notebook 1 In Colab"/>
</a>

## Part 2:
<a target="_blank" href="https://colab.research.google.com/github/Christie-Scientific-Computing/HSST_Workshop/blob/main/Part_2_head_and_neck_autoseg.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Notebook 1 In Colab"/>
</a>

## Part 3:
<a target="_blank" href="https://colab.research.google.com/github/Christie-Scientific-Computing/HSST_Workshop/blob/main/Part_3_version_changes.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Notebook 1 In Colab"/>
</a>


# TODO
    Fix colab links
    Add a quick overview to the README. Auto-segmentation + querying database. Detecting bias in a new model (part 3)  

## Background 

We will be using pre-built segmentation architecture from the [segmentation-models](https://github.com/qubvel/segmentation_models.pytorch) package. You can design your own model, but for the sake of simplicity, we will implement an existing architecture.

Pytorch can be quite intimidating, but it can be very powerful when you get to grips with it. For simplicity, we will use a wrapper around pytorch called [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/). Lightning hides a lot of the boilerplate we would typically write in pytorch, letting us very quickly and easily use best-practise methods to train our models.

## Pre-requisites:
- Google account
- Basic knowledge of Python (we will help you)

# Part 1: Abdominal auto-segmentation

In Part 1, we will show you how to build an auto-segmentation model for abdominal organs-at-risk (OARs) using pytorch and pytorch-lightning. We will be using some open data from the [MICCAI 2021 FLARE Challenge](https://flare.grand-challenge.org/). This dataset contains ~500 patients, each of which has four OARs (liver, kidneys, pancreas and spleen) segmented. This notebook serves as an introduction and will walk you through the following steps:

1. Install prerequisites and set up
2. Load data containing CT and segmentations
3. Define some preprocessing and apply it to the CT slices
4. Create a segmentation model
5. Optimise a model on the training examples
6. Test the model against the testing data
7. Visualising the results! *(You'll need to do this yourself)*

**Important: As soon as you open a notebook, you should save your own copy to your Google Drive.** This will prevent people from editing the same notebook, which will lead to chaos...

To open this notebook in colab, click this link: [colab](https://colab.research.google.com/github/Christie-Scientific-Computing/HSST_Workshop/blob/main/Part_1_abdominal_autoseg.ipynb)

# Part 2: Head and neck auto-segmentation

In Part 2, we will move sites from the abdomen to the Head and Neck (HnN). We now want **you** to train a second CNN model to segment OARs at this site. The data for this section is originally from [The Cancer Imaging Archive (TCIA)](https://github.com/deepmind/tcia-ct-scan-dataset)

The notebook provided for this part has a skeleton code provided, you'll need to fill in the rest using what you learned from Part 1.
**Important: Remember to save your own copy as soon as you click the link below!**

To open the second notebook in colab, click this link: [colab](https://colab.research.google.com/github/Christie-Scientific-Computing/HSST_Workshop/blob/main/Part_2_head_and_neck_autoseg.ipynb)

# Part 3: A new model appears!

In Part 3, you will be provided with the latest and greatest model from your auto-segmentation vendor. You will have to test the model on your data and send them very positive feedback about how great their new model is.

**Important: Remember to save your own copy as soon as you click the link below!**
To open in colab, click this link: [colab](https://colab.research.google.com/github/Christie-Scientific-Computing/HSST_Workshop/blob/main/Part_3_version_changes.ipynb)
