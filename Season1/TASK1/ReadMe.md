# Iris Dataset Exploration 🌸

## 📌 Task Objective
Explore and visualize the classic Iris dataset to understand:
- Dataset structure and composition
- Statistical properties of features
- Relationships between different flower characteristics
- Data distributions and potential outliers

## 🗃️ Dataset
**Iris Flower Dataset**  
- **Features**:
  - `sepal_length` (cm)
  - `sepal_width` (cm)
  - `petal_length` (cm) 
  - `petal_width` (cm)
- **Target**:
  - `species` (setosa, versicolor, virginica)
- **Samples**: 150
- **Source**: Built into seaborn (`sns.load_dataset('iris')`)

## 📋 Instructions Executed

### 1. Data Loading & Inspection
```python
    import seaborn as sns
    iris = sns.load_dataset('iris')
    print("Shape:", iris.shape)
    print("\nFirst 5 rows:\n", iris.head())
    print("\nInfo:\n", iris.info())
    print("\nStatistics:\n", iris.describe())