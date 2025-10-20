# 📄 kaiburr-task5-data-science: Project Alpha - Consumer Complaint Classifier 🔬

This repository contains the Python-based solution for **Task 5: Text Classification**, utilizing the Consumer Complaint Database to automatically categorize complaint narratives. The output is a robust machine learning model designed to classify raw text into specific financial complaint areas.

---

## 1. Project Objective

The primary goal is to build a machine learning model capable of classifying the free-form text found in the 'Consumer complaint narrative' field into one of the following four predefined target categories:

| ID | Category |
| :--- | :--- |
| **0** | Credit reporting, repair, or other |
| **1** | Debt collection |
| **2** | Consumer Loan |
| **3** | Mortgage |

## 2. Methodology Summary

The analysis follows a six-step Natural Language Processing (NLP) pipeline, implemented using standard Python libraries, including **Pandas**, **NLTK**, and **Scikit-learn**.

| Step | Action Summary | Key Libraries Used |
| :--- | :--- | :--- |
| **1. EDA & Feature Eng.** | Loaded and filtered data to define the target variable and features (Complaint Narrative). | `Pandas`, `Scikit-learn` (`LabelEncoder`) |
| **2. Text Pre-processing** | Cleaning operations including lower-casing, punctuation removal, tokenization, and Stop Word elimination. | `NLTK` (`Stopwords`, `Word_tokenize`) |
| **3. & 4. Modeling** | Creation of a robust classification `Pipeline` combining feature extraction (e.g., TF-IDF) and the chosen classifier. | `Scikit-learn` (`Pipeline`, Feature Extraction Modules) |
| **5. Model Evaluation** | Training a suitable classifier (e.g., `LinearSVC` or `Logistic Regression`) and evaluating performance using standard metrics. | `Scikit-learn` |
| **6. Prediction** | Demonstration of the final model's ability to accurately classify unseen text input. | `Scikit-learn` |

## 3. Installation & Run Instructions

### Prerequisites
1. **Download Data:** Obtain the Consumer Complaint Database (CSV file) and place it in the project root directory.
2. **Install Libraries:** Ensure the necessary Python dependencies are installed.

```bash
pip install pandas numpy scikit-learn nltk
```

### Running the Analysis
1. Ensure all prerequisites are met (data file present and libraries installed).
2. Execute the cells in the `Text_Classification.ipynb` Jupyter Notebook sequentially. The notebook handles data loading, preprocessing, model training, and final evaluation.

## 4. Model Evaluation & Prediction Proofs (Mandatory Verification)

The following image section verifies the model's predictive capability and performance on the held-out test set.

| Verification Focus | Proof Description | Proof Screenshot |
| :--- | :--- | :--- |
| **Model Evaluation** | Shows the Classification Report (Precision, Recall, F1-score) proving effective multi-class performance. | *[Insert Proof Screenshot 1 Here]* |
| **Sample Prediction** | Demonstrates the model successfully classifying a new, unseen complaint narrative into one of the four categories. | *[Insert Proof Screenshot 2 Here]* |

***
*Note: Please insert the composite verification image showing the classification report and a sample prediction where indicated above.*
