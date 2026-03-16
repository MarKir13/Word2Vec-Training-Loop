# Word2Vec Training Implementation

This project is a custom implementation of Word2Vec model using the Skip-gram architecture. Whole project is based only on NumPy library without using any high-level Machine Learning libraries.

## Key Features
* **Skip-Gram Architecture:**  Model learns to predict surrounding context words based on center target word
* **Pure NumPy operations:** All operations including whole training loop are implemented with NumPy matrix operations
* **Model Persistence:** Features a model saving system using `.npz` file. This allows user to pause and resume the learning process across different sessions without losing progress

## Technical Details
* **Dataset:** Text8
* **Optimization:** Stochastic Gradient Descent
* **Loss Function:** Categorical Cross-Entropy (with Softmax)