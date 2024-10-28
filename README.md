# A Privacy-Preserving Data Augmentation Approach for Credit Card Fraud Detection

This project implements a privacy-preserving data augmentation approach using vertical federated learning for credit card fraud detection. The code demonstrates how to apply CKKS and Paillier homomorphic encryption schemes to secure the training process. The project uses the **credit card fraud dataset** from Kaggle and applies the **SMOTE** technique to augment the minority class data.

## Dataset

The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It is split into a training set and a test set, available from the following links:
- **Training Set**: [Download from Google Drive](https://drive.google.com/file/d/1rJlgEOpakousK-83fKKNg9xrjfPvU8sf/view)
- **Test Set**: [Download from Google Drive](https://drive.google.com/file/d/1nnRE2v7J-zt5xyR9dy9QCwE1cShkDKH9/view)

## Experiment Steps

### Step 1: Download and Prepare the Data
- Download the **training** and **test datasets** from the links above.
- Place the datasets in the **root directory** of the project.

### Step 2: Generate the SMOTE Dataset
- Run the following command to generate the SMOTE-augmented training dataset:
  ```bash
  python project_smote.py

- This will create the file creditcard_train_SMOTE_1.csv, which will be used as the training dataset for the next step.

### Step 3: Run the Experiment
#### CKKS Encryption Scheme
- To run the experiment with the CKKS homomorphic encryption scheme, use the following command:
  ```bash
  python p_e.py --he_scheme=ckks --dataset=creditcard_train_SMOTE_1.csv

#### Paillier Encryption Scheme
- To run the experiment with the Paillier  encryption scheme, use the following command:
  ```bash
  python p_e.py --he_scheme=paillier --dataset=creditcard_train_SMOTE_1.csv

## Experiment Results


| Table 1 Average Computational Time for Different Encryption Methods |
|----------------------------------|------------------|--------------------------|-----------------------------|
| **Encryption Methods**           | No Encryption   | Encryption with CKKS     | Encryption with Paillier    |
| **Time(s)**                      | 124.0 (9.2)     | 126.0 (8.5)              | 6626.1 (57.1)               |


Table 4 presents the average computational costs for various encryption methods. The first value indicates the average computational time, and the value in parentheses indicates the variance. The results show that CKKS introduces only a 6% overhead in time compared to the unencrypted scenario, with an average time of 126.0 seconds—just 2 seconds more than the non-encrypted baseline of 124.0 seconds. In contrast, Paillier encryption incurs substantially higher overhead, averaging 6626.1 seconds. The efficiency of CKKS arises from its support for vectorized operations, whereas Paillier lacks this capability, relies on sequential processing, and does not natively support floating-point data types.

