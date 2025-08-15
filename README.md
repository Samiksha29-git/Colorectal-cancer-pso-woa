Enhanced Colorectal Cancer Diagnosis: A Hybrid Vision Transformer with Bio-Inspired Optimization Project Overview 

A complete machine learning pipeline exists within this project to classify histopathological images from colorectal cancer tissue. A hybrid deep learning model which combines BEiT with EfficientNet achieves high accuracy when classifying Kather_texture_2016 dataset images.
The project extends basic training functionality through the implementation of two meta-heuristic optimization algorithms which include Particle Swarm Optimization (PSO) and Whale Optimization Algorithm (WOA). The model undergoes hyperparameter optimization through these algorithms which also tests their results against manually chosen baseline model parameters. The method proves effective for building a classification model with optimal performance and model parameters.
The project development was conducted by Prathana Sharma, Samiksha Sandeep Zokande and Dr. Hemanth K S.

Features 

Data Pipeline: The robust modular pipeline manages data copying along with folder renaming and preprocessing steps  including histogram equalization and blurring and resizing.
Dataset Splitting: The processed dataset gets automatically divided  by the pipeline into train and validation and test sets.
Hybrid Deep Learning Model: The custom  HybridBeitEffNet model utilizes Vision Transformer and CNN components to achieve better performance in deep learning tasks.
Hyperparameter Optimization: The model employs both PSO and WOA algorithms to discover the best  hyperparameters for its final layer architecture.

Comprehensive Evaluation: The model undergoes a detailed performance evaluation  through standard metrics which includes Accuracy, F1-Score and visual displays of both Confusion Matrix and  ROC curves.
Modular Code: The project features organized functions that are reusable and provide straightforward pathways  for editing and extending the code base.
Dataset 

The project uses the Kather_texture_2016_image_tiles_5000 dataset, which consists of 5,000 colorectal cancer histology image tiles. The images are classified into eight distinct texture classes: Tumor, Stroma, Complex, Lympho, Debris, Mucosa, Adipose, and Empty.

Source: https://zenodo.org/records/53169#.W6HwwP4zbOQ

Dependencies 
To run this project, you will need the following libraries. You can install them using pip:
pip install torch torchvision torchaudio timm numpy pandas scikit-learn matplotlib seaborn tqdm opencv-python
We recommend using a GPU-enabled environment (e.g., Google Colab, Kaggle, or a local machine with CUDA) for efficient training.

Usage 

The project is designed to run as a single Python script. The main workflow is defined in the main() function, which sequentially executes all steps of the pipeline, from data preparation to model evaluation.
The main() function will perform the following steps automatically:
Copy and Preprocess the dataset.
Split the data.
Train a Base Model with default hyperparameters.
Run PSO optimization, retrain the model with the best hyperparameters, and evaluate.
Run WOA optimization, retrain the model with the best hyperparameters, and evaluate.
Print a final summary of all three models' performance.

Results 

The script generates plots and prints detailed reports for each model. Key outputs include:
Confusion Matrix: A visual representation of the model's predictions on the test set.
ROC Curve: A plot showing the model's performance across different classification thresholds for each class.
Loss and Accuracy curves: Plots presenting model’s accuracy and loss across all epochs
Classification Report: A detailed text report with Precision, Recall, and F1-Score for each class.
Final Summary: A comparative table of the performance metrics (e.g., accuracy, loss, F1-score) for the Base, PSO-optimized, and WOA-optimized models.

Contribution and Collaboration 

This is a group project developed by Samiksha Sandeep Zokande, Prathana Sharma and Dr. Hemanth K S. We have consolidated our work into a single script to maintain a clean and unified codebase. We welcome suggestions and improvements!
If you'd like to contribute, feel free to fork the repository and submit a pull request. We can discuss new features, bug fixes, or improvements to the optimization algorithms.

