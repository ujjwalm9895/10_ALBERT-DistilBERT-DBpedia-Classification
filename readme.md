# DBpedia Multi-class classification using ALBERT and DistilBERT

## Business Objective
This project focuses on addressing the shortcomings of existing models in terms of memory optimization, prediction latency, and space usage. The project also introduces two new models:
- ALBERT: A Lite BERT for Self-supervised Learning of Language Representations.
- DistilBERT: A distilled version of BERT: smaller, faster, cheaper, and lighter. 

The goal is to optimize space and memory usage with minimal impact on prediction accuracy.

---

## Data Description
The project uses the DBpedia ontology classification dataset, which consists of 14 non-overlapping classes selected from DBpedia 2014. From each class, 40,000 training samples and 5,000 testing samples are randomly chosen, resulting in a training dataset of 560,000 and a testing dataset of 70,000 samples. The dataset includes three columns:
- Title: The title of the document.
- Content: The body of the document.
- Label: One of 14 possible topics.

---

## Aim
The project aims to build two classification models, ALBERT and DistilBERT, for the DBpedia ontology dataset.

---

## Tech Stack
- **Language**: `Python`
- **Libraries**: `datasets`, `numpy`, `pandas`, `matplotlib`, `ktrain`, `transformers`, `tensorflow`, `sklearn

---

## Approach
1. Install the required libraries.
2. Load the 'DBpedia' dataset.
3. Load train and test data.
4. Data pre-processing:
   - Assign column names to the dataset.
   - Append and save the dataset.
   - Drop redundant columns.
   - Add a text length column for visualization.
5. Perform data visualization:
   - Histogram plots.
6. ALBERT model:
   - Check for hardware and RAM availability.
   - Import necessary libraries.
   - Data interpretations.
   - Create an ALBERT model instance.
   - Split the train and validation data.
   - Perform data pre-processing.
   - Compile ALBERT model in a K-train learner object.
   - Fine-tune the ALBERT model on the dataset.
   - Check performance on validation data.
   - Save the ALBERT model.
7. DistilBERT model:
   - Check for hardware and RAM availability.
   - Import necessary libraries.
   - Data interpretations.
   - Create a DistilBERT model instance.
   - Split the train and validation data.
   - Perform data pre-processing.
   - Compile DistilBERT model in a K-train learner object.
   - Fine-tune the DistilBERT model on the dataset.
   - Check performance on validation data.
   - Save the DistilBERT model.
8. Create a BERT model using the DBpedia dataset for comparative study.
9. Follow the above steps for creating a BERT model on the 'Emotion' dataset.
10. Follow the above steps for creating an ALBERT model on the 'Emotion' dataset.
11. Follow the above steps for creating a DistilBERT model on the 'Emotion' dataset.
12. Save all the generated models.

---

## Modular Code Overview

1. **Input**: Contains data for analysis, including CSV files and a tar.gz file.
   - dbpedia_14_test.csv
   - dbpedia_14_train.csv
   - dbpedia_csv.tar.gz

2. **Src**: The most important folder with modularized code for all the project steps. It includes:
   - `Engine.py`
   - `ML_Pipeline`: A folder with functions split into different Python files, appropriately named. These functions are called within `Engine.py`.

3. **Output**: Contains the ALBERT and DistilBERT models trained on this data. These models can be easily loaded and used for future applications without retraining.
   - Note: These models are built on a subset of the data. To obtain models for the entire dataset, run `Engine.py` using the complete data for training.

1. **Lib**: A reference folder with original IPython notebooks

---

## Getting Started

1. Install the required packages stated in the `requirements.txt` file:
   - For Anaconda:
     ```shell
     conda create --name <yourenvname>
     conda activate <yourenvname>
     pip install -r requirements.txt
     ```
   - For Python Interpreter:
     ```shell
     pip install -r requirements.txt
     ```

2. The repository is modularized into individual sections performing specific tasks.
   - Navigate to the `src` folder.
   - Under `src`, you'll find:
     - **ML_Pipeline**: Contains modules with function declarations for specific machine learning tasks.
     - **engine.py**: The core of the project where all function calls are made.

3. Run or debug the `engine.py` file, and all necessary steps will be executed automatically based on the logic.

4. Input datasets are stored in the `input` folder.

5. Predictions and models are stored in the `output` folder.

---
