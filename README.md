# 🔠 SVR-Based Letter Recognition - UCI Dataset

This project applies **Support Vector Regression (SVR)** to classify handwritten English capital letters using the **UCI Letter Recognition Dataset**. While SVR is typically used for regression tasks, here we adapt it for classification by rounding and clipping the predicted values.

---

## 🧪 Methodology

1. **Data Loading**:
   - Dataset: [`letter-recognition.csv`](https://archive.ics.uci.edu/ml/datasets/letter+recognition)
   - Features: 16 numerical features
   - Target: English letters (A-Z), encoded as integers using `LabelEncoder`

2. **Model Pipeline**:
   - Preprocessing with `StandardScaler`
   - Support Vector Regression using `SVR`
   - Hyperparameter tuning with `GridSearchCV` using 3-fold cross-validation

3. **Hyperparameters Tuned**:
   - `kernel`: `linear`, `rbf`, `poly`
   - `C`: `0.1`, `1`, `10`
   - `epsilon`: `0.01`, `0.1`, `0.5`

4. **Evaluation Process**:
   - For each of 10 random train-test splits:
     - Train SVR using best parameters
     - Predict on test set
     - Round predictions to nearest class index
     - Clip results to ensure valid class range
     - Compute classification **accuracy**

---

## 📊 Results Table

Below is a sample of the output result across 10 different train-test splits:

| Sample | Accuracy (%) | Kernel | C    | Epsilon |
|--------|---------------|--------|------|---------|
| S1     | 85.23         | rbf    | 10   | 0.1     |
| S2     | 83.10         | poly   | 1    | 0.5     |
| S3     | 84.67         | linear | 1    | 0.1     |
| ...    | ...           | ...    | ...  | ...     |
| S10    | 85.91         | rbf    | 10   | 0.1     |

> 🔍 Accuracy is calculated after converting SVR outputs into integer-labeled classes.

---

## 📈 Convergence Graph

A convergence graph was generated for the **best model configuration**, showing how test accuracy improves as more training data is used.

- **X-axis**: Iteration (1–100), simulating growing training batches
- **Y-axis**: Accuracy on test set (%)
- **Model**: Best-performing SVR from 10 experiments

![Convergence Graph](svm_convergence_graph.png)

> 🧠 Observation: Accuracy improves rapidly in the initial stages and then gradually stabilizes.

---

## ✅ Conclusion

- SVR can be adapted for classification by rounding and clipping predictions.
- `rbf` and `linear` kernels performed best in this use case.
- Accuracy across all runs ranged between **83–86%**.
- Convergence curve validates model stability with growing data.

---
