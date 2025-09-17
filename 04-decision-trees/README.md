
# Decision Trees - Classification with Iris Dataset

## Project Overview

This project demonstrates how to build, train, and visualize a decision tree model using the **Iris dataset**. The decision tree is a supervised machine learning model used for both classification and regression tasks. In this example, the tree is built to classify Iris flower species based on four features: sepal length, sepal width, petal length, and petal width.

The project is split into two main files:
- **`decision_trees.ipynb`**: Jupyter notebook that walks through the decision tree theory, implementation, and visualizations.
- **`main.py`**: Python script containing a `main()` function for running the decision tree classification on the Iris dataset.

## Project Files

### 1. `decision_trees.ipynb`

The Jupyter notebook provides an interactive tutorial that covers:
- **Theory of Decision Trees**: Explanation of the Gini Impurity, Entropy, Information Gain, and Mean Squared Error (MSE) for classification and regression tasks.
- **Decision Tree Implementation**: Code that loads the Iris dataset, splits the data into training and testing sets, trains a decision tree classifier, and evaluates the model.
- **Visualization**: The decision tree is visualized to help understand how the model splits the data at each node.

### 2. `main.py`

The Python script is the standalone version of the notebook and can be run directly from the command line or terminal. It contains a `main()` function which:
- Loads the Iris dataset.
- Splits the data into training and testing sets.
- Trains a decision tree classifier using the Gini Impurity criterion.
- Makes predictions and evaluates the model's accuracy.
- Visualizes the decision tree structure.
- Prints the confusion matrix and classification report.

## Getting Started

### Prerequisites

To run this project, you need the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

You can install them using `pip`:

```bash
pip install numpy pandas matplotlib scikit-learn
```

Alternatively, you can use the provided `requirements.txt` to install the necessary dependencies:

```bash
pip install -r requirements.txt
```

### Running the Code

#### 1. Running the Jupyter Notebook (`decision_trees.ipynb`)

1. Make sure you have **Jupyter Notebook** installed. If not, you can install it via `pip`:

    ```bash
    pip install notebook
    ```

2. Launch Jupyter Notebook:

    ```bash
    jupyter notebook
    ```

3. Open the `decision_trees.ipynb` file from the Jupyter interface and run the cells sequentially.

#### 2. Running the Python Script (`main.py`)

1. Ensure you have the required libraries installed, as mentioned in the **Prerequisites** section.
2. Run the `main.py` script from the terminal or command line:

    ```bash
    python main.py
    ```

    This will execute the decision tree classification, print the accuracy, confusion matrix, classification report, and display the decision tree visualization.

## Key Concepts

### 1. **Decision Tree Theory**
- **Gini Impurity** and **Entropy** are the primary splitting criteria for classification tasks.
- **Mean Squared Error (MSE)** is commonly used for regression tasks.
- The decision tree recursively splits the data based on these criteria to create an easy-to-understand classification or regression model.

### 2. **Model Evaluation**
- The performance of the decision tree is evaluated using **accuracy**, **confusion matrix**, and **classification report**.
- **Visualization** helps you understand how the tree splits the data and makes predictions.

## Conclusion

This project demonstrates the power and simplicity of decision trees for classification tasks using the Iris dataset. You can modify and expand the code for more complex datasets and real-world applications. By understanding how the decision tree works and how to visualize it, you can gain insights into model decision-making.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
