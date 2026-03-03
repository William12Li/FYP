import numpy as np

class WHFramework:
    def __init__(self, L=10):
        """
        Initialize the framework.
        :param L: Memory depth or the number of columns in the convolution matrix.
        """
        self.L = L
        self.weights = None

    def convmtx(self, x, L):
        """
        Generate the convolution matrix (based on previous logic implementation).
        """
        N = len(x)
        X = np.zeros((N, L))
        for i in range(L):
            # Fill in the shifted data
            X[i:, i] = x[:N-i]
        return X

    def preprocess(self, x_input):
        """
        Data preprocessing: Convert the input vector into a feature matrix.
        """
        return self.convmtx(x_input, self.L)

    def train(self, x_input, y_target):
        """
        Training logic: Using Least Squares as an example here.
        """
        X = self.preprocess(x_input)
        # Solve the linear system: X * w = y_target
        self.weights, _, _, _ = np.linalg.lstsq(X, y_target, rcond=None)
        print("Framework: Model training complete.")

    def predict(self, x_input):
        """
        Prediction logic.
        """
        if self.weights is None:
            raise ValueError("Model has not been trained yet!")
        X = self.preprocess(x_input)
        return np.dot(X, self.weights)

    def calculate_nmse(self, y_true, y_pred):
        """
        Calculate NMSE (Normalized Mean Square Error), a feature mentioned in your symbols list.
        """
        mse = np.mean(np.abs(y_true - y_pred)**2)
        power = np.mean(np.abs(y_true)**2)
        return 10 * np.log10(mse / power)

# Quick test code
# Replace the code at the bottom of whframework.py with this:
if __name__ == "__main__":
    # 1. Initialize the framework
    framework = WHFramework(L=5) 
    print("--- WH Framework Random Data Test Started ---")

    # 2. Generate random Gaussian data
    # np.random.randn generates Gaussian distributed data with mean 0 and variance 1
    n_samples = 100
    x_random = np.random.randn(n_samples) 
    
    # Simulate a real system: Assume output signal is 1.5 times the input plus some random noise
    y_target = 1.5 * x_random + np.random.randn(n_samples) * 0.1

    print(f"Generated {n_samples} random samples.")
    print(f"First 5 input samples: {x_random[:5]}")

    # 3. Run preprocessing (generate convolution matrix)
    X_matrix = framework.preprocess(x_random)
    print(f"Convolution matrix shape: {X_matrix.shape}") # Should be (100, 5)

    # 4. Run training (Least Squares)
    framework.train(x_random, y_target)
    print(f"Trained model weights: \n{framework.weights}")

    # 5. Validation: Predict on the same batch of data
    y_pred = framework.predict(x_random)
    
    # 6. Print comparison of the first 10 predicted vs. actual values
    print("Actual (Target) | Predicted (Pred)")
    for t, p in zip(y_target[:10], y_pred[:10]):
        print(f"{t:14.4f} | {p:14.4f}")

    # 7. Calculate NMSE error
    nmse_val = framework.calculate_nmse(y_target, y_pred)
    print(f"\nFinal test Normalized Mean Square Error: {nmse_val:.2f} dB")