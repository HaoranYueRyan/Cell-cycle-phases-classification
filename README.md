# Cell-cycle-phases-classification

Use deep learning methods to identify label-free nucleus for cell cycle phases classification.

Cell cycle mechanism plays a major part in cancer development and understanding cell cycle mechanisms are crucial steps for cancer treatment. In this project, we have optimised an already published deep learning based segmentation model, Cellpose, for image segmentation. Then we have developed a pipeline to allow accurate classification of cell phase cycles based on one fluorescence channel (larter will upload two channel as comparison). From this project, we are able to provide a label free estimation of the cell cycle phase based on the morphology of the cytoplasm in the high throughput analysis of cell cycle changes and free up the remaining imaging channel for the investigation of additional cellular states.

# Genenate the training data
 
 In this pipeline, the imaging data were collected from Dr Helfrid Hochegger’s laboratory. All images were stored on an OMERO server (OMERO 5.7.0) and accessed via the Opero-Python API. The data_generate.ipynb can generate the data set consist of 4 phase cell cycle, G1, S, G2, M.
 
 # Training the model 
 
 Here, we have tained the CNN, ResNet and Efficeint models with the dataset. By comparing these models, the ResNet model has better apperance. All the models and traing py files have upload on this project.

# Dependencies

Windows and Mac Os are supported for running the code. For training the model with MPS, you will need a Mac Os with M1. All the code has been steady tested on MAC Os.

For the segmentation part, you can install the Cellpose with GUI call `python -m pip install 'cellpose[gui]', which can easily manual label the cell to get more pricesly cellpose model. 
