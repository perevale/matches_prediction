# Predicting sports matches with neural models”
This repository provides a codebase for a master’s thesis “Predicting sports matches with neural models”.

The models can be created from a ```scripts/main.py``` file. 
The functions available include:

 - ```run_flat_cont```: for creating and training a Simple ANN model with embedding;
 - ```run_gnn_cont```: for creating and training a GNN model with embedding;
 - ```run_exist_model```: for running and testing an existing model;
 - ```calculate_rps```: for computing RPS score for an existing model;
 - ``confusion_matrix```: for computing a confusion matrix on the testing data for an existing model.

For easy testing the functionality some quick start commands are predefined in the main condition:
 - 5: ```run_gnn_cont```
 - 8: ```run_flat_cont```
 - 11: ```run_exist_model```
 - 12: ```confusion_matrix```
 - 13: ```calculate_rps```


The user has to check the following parameters before running the functions:

 - function_id : id of the executed function 
 - exp_num : number of experiment (when applicable)
 - dataset_filename : the name of the input training data set with the relative path (when applicable)   
 - model_filename : the name of the existing model with the relative path


Some of the functions (marked under "Unused functions") are saved from previous experiments, are not updated and should not be taking into consideration when reviewing the code.

The requirements to run the project are listed in the ```requirement.txt```.

The instructions to the install the PyTorch Geometric after the installation of a new conda environment:

    pip install pytorch  
    pip install torch-scatter==latest+cpu torch-sparse==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.6.0.html  
    pip install torch-geometric


