# RSNA-Intracranial-Hemorrhage-Detection


Healthcare is ripe for innovation, with every advancement in machine learning speeding it along. One area in particular, medical imaging, is rapidly evolving with each stepwise improvement in neural network techniques. In this blog, I will discuss my capstone project for Flatiron School’s Data Science bootcamp, in which I use a convolutional neural networks to detect intracranial hemorrhages, or bleeding in and around the brain. 

 

First and foremost, though, a huge thank you to Allunia and her Kaggle notebook, which I drew from as my baseline, as well as all the other Kagglers who helped piece together the functions and solutions to common problems. In a future post, I would like to talk about my initial experiment doing this project blind -- i.e. not looking at any others' code -- and how I learned a ton from that process. However, I learned even more from all the brilliant data scientists putting their code out in the public and for that, I'm very grateful. 


## Data

The project originated from a Kaggle competition. The data came in DICOM (.dcm) files. DICOM is the Digital Imaging and Communications in Medicine standard for biomedical images. It specifies a data interchange protocol, a digital image format (for CT scans in our case), and file structures.

 

In other words, each CT scan was not only a large 512x512 pixel image, but a host of metadata as well. These files, consequently, were too large to load on my computer. For this project, I decided to use cloud computing as I could download the Kaggle files there and more efficiently run deep-learning models with the aid of rented GPUs. After comparing pricing and ease-of-setup, I chose to run my project on Paperspace over AWS, Google Cloud, and others. 

 

I successfully obtained 103,772 files to use for my project. I used the pydicom library to access and manipulate the files. The goal was to identify the probability of each of five subtypes of hemorrhage existing in the image, as well as a catch-all 'any' category. Each image ID corresponds to exactly one DICOM file and 6 observations in the labeled data frame provided through Kaggle. 

# Methodology

A. Preprocessing

building custom functions

error-handling – expect loads of errors

many parameters to change for testing different models

 

B. Build and run a convolutional neural network

using TensorFlow and Keras modules

multi-class multi-label classification problem

transfer learning (ResNet, VGG)

GPU from cloud computing platform will speed up process greatly