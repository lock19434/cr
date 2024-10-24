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

![Comparing the Cost of Encryption: Paillier, CKKS, and No Encryption](./image/Comparing%20the%20Cost%20of%20Encryption.png)

The figure shows a comparison of training time between Paillier, CKKS, and no encryption. As can be seen, the training time for the no encryption approach is almost zero, CKKS encryption slightly increases the time but still remains close to the no encryption scheme, while Paillier encryption significantly increases the training time to around 6000 seconds.

**Why is there such a large gap between CKKS and Paillier?**

The main difference between CKKS and Paillier lies in how they handle data. CKKS is a homomorphic encryption scheme that supports vectorized operations, meaning it can encrypt and compute on an entire batch of data simultaneously. This significantly boosts efficiency. During homomorphic computation, CKKS allows for addition, multiplication, and other operations directly on encrypted vectors, resulting in relatively low computational overhead.

On the other hand, Paillier homomorphic encryption can only perform addition on single encrypted values and does not support vectorized operations. Every operation requires processing each individual ciphertext, which leads to a much higher computational cost, especially when handling large datasets.

Thus, CKKS's ability to support vector operations and its efficiency in computation help keep the time cost low while ensuring data privacy, whereas Paillier's limitations result in a significant increase in computation time.

