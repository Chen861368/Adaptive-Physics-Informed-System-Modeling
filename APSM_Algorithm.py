# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import svd
from scipy.linalg import fractional_matrix_power
np.random.seed(0)
import time as tm 
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch

def plot_y_nmse_over_iterations(y_nmse_tests):
    """
    Plot y_nmse_test over iterations and display the plot directly.

    Parameters:
    - y_nmse_tests: List or array of y_nmse_test values over iterations
    """
    # Set global font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'

    # Plot y_nmse_test over iterations
    plt.figure(figsize=(12, 8), dpi=300)
    plt.plot(range(1, len(y_nmse_tests) + 1), y_nmse_tests, marker='o', linestyle='-', color='blue', linewidth=2.5, label='NMSE over Iterations')

    # Customize plot labels and appearance
    plt.rcParams['font.size'] = 15
    plt.xlabel('Iteration', fontsize=25)
    plt.ylabel('NMSE', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle='--', linewidth=0.5)

    # Add legend and adjust its appearance
    legend = plt.legend(fontsize='x-large', handlelength=2, edgecolor='black', frameon=True, fancybox=False)
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_edgecolor('black')

    # Customize plot borders
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


def compare_psd(y, y_est, fs=50.0):
    """
    Compare the Power Spectral Density (PSD) of original data and estimated data with enhanced readability and aesthetics.
    Displays the plots directly.

    Parameters:
    y (numpy.ndarray): The matrix of original responses (outputs) with shape (time_steps, n_outputs).
    y_est (numpy.ndarray): The matrix of estimated responses (outputs) with shape (time_steps, n_outputs).
    fs (float): The sampling frequency of the data.

    Returns:
    None
    """
    # Set the style and context
    sns.set(style="whitegrid", context="talk")
    
    # Set global font to Times New Roman and increase size
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16
    
    n_outputs = y.shape[1]
    
    for i in range(n_outputs):
        fig, ax = plt.figure(figsize=(12, 8), dpi=300), plt.gca()
        
        # Compute PSD for original data
        f, Pxx_orig = welch(y[:, i], fs, nperseg=1024)
        ax.semilogy(f, Pxx_orig, label=f'Original Output {i+1}', color='b', linewidth=1.5)
        
        # Compute PSD for estimated data
        f, Pxx_est = welch(y_est[:, i], fs, nperseg=1024)
        ax.semilogy(f, Pxx_est, label=f'Estimated Output {i+1}', color='r', linestyle='--', linewidth=1.5)
        
        # Customize plot labels and appearance
        plt.xlabel('Frequency (Hz)', fontsize=25)
        plt.ylabel('Power Spectral Density (m²/s⁴/Hz)', fontsize=25)
        plt.legend(fontsize='x-large', handlelength=2, loc='upper right', frameon=True)
        plt.grid(True, linestyle='--')
        
        # Set tick parameters
        plt.tick_params(axis='both', which='major', labelsize=20)
        
        # Change the border color to black
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        
        # Ensure layout fits well
        plt.tight_layout()
        
        # Display the plot directly
        plt.show()




def visualize_results_monit(time, actual_data, estimated_data, n, iterations):
    """
    Visualize the true and estimated states with stylistic adjustments for enhanced readability
    and aesthetics. Displays the plots directly instead of saving them as PDFs.

    Parameters:
    - time: Time steps
    - actual_data: True displacements
    - estimated_data: Estimated states
    - n: Number of degrees of freedom
    - iterations: Number of iterations
    """
    sns.set(style="white", context="talk")  # Use a white background and context suitable for presentations

    for i in range(n):
        plt.figure(figsize=(12, 8), dpi=300)  # Set figure size and DPI

        # Set global font to Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 16

        # Plot actual data with a solid blue line
        plt.plot(time[:iterations], actual_data[i, :iterations], label=f'Monitoring data', linestyle='-', color='blue', linewidth=2.5)

        # Plot estimated data with a dashed orange line
        plt.plot(time[:iterations], estimated_data[i, :iterations], label=f'Predicted data', linestyle='--', color='orange', linewidth=2.5)

        # Customize labels with explicit font sizes
        plt.xlabel('Time (seconds)', fontsize=25)
        plt.ylabel('Acceleration (m/s²)', fontsize=25)

        # Enlarge legend line and text, add a black edge to the legend with a white background for visibility
        legend = plt.legend(fontsize='x-large', handlelength=2, edgecolor='black', loc='upper right', frameon=True, fancybox=False)
        # Set the linewidth of the legend border
        legend.get_frame().set_linewidth(1.5)

        # Explicitly set tick label sizes
        plt.tick_params(axis='both', which='major', labelsize=20)

        # Use dashed grid lines for better readability
        plt.grid(True, linestyle='--', linewidth=0.5)

        # Ensure layout fits well
        plt.tight_layout()

        # Display the plot directly
        plt.show()










def evaluate_predictions(actual, predicted):
    """
    Evaluate the quality of predictions using statistical metrics: MSE, NMSE, and R².
    Handles complex data by evaluating only the real part.

    Parameters:
    - actual (np.ndarray): The actual data (ground truth), which can include complex values.
    - predicted (np.ndarray): The predicted data, which can include complex values.

    Returns:
    - mse (float): Mean Squared Error of the predictions, a measure of the average squared difference between actual and predicted values.
    - nmse (float): Normalized Mean Squared Error of the predictions, calculated as MSE divided by the variance of the actual data.
    - r2 (float): Coefficient of Determination (R²), indicating the proportion of variance in the actual data explained by the predictions.
    """
    # Extract the real part of the actual and predicted data
    actual_real = np.real(actual)
    predicted_real = np.real(predicted)
    
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(actual_real, predicted_real)
    
    # Calculate Normalized Mean Squared Error (NMSE)
    variance_actual = np.var(actual_real)  # Variance of the actual data
    nmse = mse / variance_actual
    
    # Calculate Coefficient of Determination (R²)
    r2 = r2_score(actual_real, predicted_real)
    
    return mse, nmse, r2

def kalman_filter_update(A, B, C, Q, R, x_hat, P, u, y):
    """
    Perform one iteration of the Kalman filter update, which includes both the prediction
    and the update step.

    Parameters:
    A (numpy.ndarray): State transition matrix
    B (numpy.ndarray): Control input matrix
    C (numpy.ndarray): Observation matrix
    Q (numpy.ndarray): Process noise covariance matrix
    R (numpy.ndarray): Measurement noise covariance matrix
    x_hat (numpy.ndarray): Initial state estimate
    P (numpy.ndarray): Initial estimate covariance matrix
    u (numpy.ndarray): Control input vector
    y (numpy.ndarray): Observation vector

    Returns:
    tuple: Updated state estimate (x_hat), updated estimate covariance (P), and predicted observation (C @ x_hat_pred)
    """

    def predict(x_hat, P, A, B, u, Q):
        """
        Prediction step of the Kalman filter.
        Predicts the next state and updates the covariance matrix.

        Parameters:
        x_hat (numpy.ndarray): Current state estimate
        P (numpy.ndarray): Current estimate covariance matrix
        A (numpy.ndarray): State transition matrix
        B (numpy.ndarray): Control input matrix
        u (numpy.ndarray): Control input vector
        Q (numpy.ndarray): Process noise covariance matrix

        Returns:
        tuple: Predicted state estimate (x_hat_pred) and predicted covariance matrix (P_pred)
        """
        x_hat_pred = A @ x_hat + B @ u
        P_pred = A @ P @ A.T + Q
        return x_hat_pred, P_pred

    def update(x_hat_pred, P_pred, y, C, R):
        """
        Update step of the Kalman filter.
        Updates the state estimate and the covariance matrix based on the observation.

        Parameters:
        x_hat_pred (numpy.ndarray): Predicted state estimate
        P_pred (numpy.ndarray): Predicted covariance matrix
        y (numpy.ndarray): Observation vector
        C (numpy.ndarray): Observation matrix
        R (numpy.ndarray): Measurement noise covariance matrix

        Returns:
        tuple: Updated state estimate (x_hat) and updated covariance matrix (P)
        """
        S = C @ P_pred @ C.T + R
        K = P_pred @ C.T @ np.linalg.inv(S)  # Kalman gain
        x_hat = x_hat_pred + K @ (y - C @ x_hat_pred)
        P = (np.eye(len(P_pred)) - K @ C) @ P_pred
        return x_hat, P

    # Perform prediction
    x_hat_pred, P_pred = predict(x_hat, P, A, B, u, Q)

    # Perform update
    x_hat, P = update(x_hat_pred, P_pred, y, C, R)

    return x_hat, P, C @ x_hat_pred


def gradient_descent_update(A_k, x_k, y_k, learning_rate):
    """
    Perform one iteration of gradient descent to update a cyclic matrix A_k
    based on a single column vector x_k and the corresponding target value y_k.

    Parameters:
    A_k (numpy.ndarray): Current matrix to be updated
    x_k (numpy.ndarray): Input vector
    y_k (float): Target value corresponding to x_k
    learning_rate (float): Learning rate for gradient descent

    Returns:
    numpy.ndarray: Updated matrix A_k
    """
    # Compute gradient
    grad = -2 * (y_k - A_k.dot(x_k)) * x_k.T
    
    # Update A_k using the gradient
    A_k -= learning_rate * grad
    
    return A_k

def create_hankel_matrix(data, rows, cols):
    """
    Create a Hankel matrix from the given data.

    Parameters:
    data (numpy.ndarray): The input data matrix with shape (n_samples, n_features).
    rows (int): The number of rows in the Hankel matrix.
    cols (int): The number of columns in the Hankel matrix.

    Returns:
    hankel_matrix (numpy.ndarray): The resulting Hankel matrix.
    """
    n_samples, n_features = data.shape
    hankel_matrix = np.zeros((cols, rows* n_features))

    for i in range(cols):
        for j in range(rows):
            if i + j < n_samples:
                hankel_matrix[i, j * n_features: (j + 1) * n_features] = data[i + j]

    return hankel_matrix.T


# @jit(nopython=True)  # 加速Hankel矩阵的生成
def era_analysis(y, rows, cols, num_in, num_out, system_order):
    """
    Perform Eigensystem Realization Algorithm (ERA) analysis on the given data.

    Parameters:
    y (numpy.ndarray): The matrix of system responses (outputs) with shape (n_samples, n_features).
    rows (int): The number of rows in the Hankel matrix.
    cols (int): The number of columns in the Hankel matrix.
    system_order (int): The order of the system to be identified.

    Returns:
    A (numpy.ndarray): The system matrix A.
    B (numpy.ndarray): The input matrix B.
    C (numpy.ndarray): The output matrix C.
    D (numpy.ndarray): The feedthrough matrix D.
    """
    # Create Hankel matrices
    num_samples, num_sensors = y.shape
    
    H = create_hankel_matrix(y, rows+1, cols)
    H0=H[:rows * num_sensors,:]
    H1=H[num_sensors:,:]
    
    # H0 = create_hankel_matrix(y, rows, cols)
    # H1 = create_hankel_matrix(y[1:], rows, cols)

    # Perform SVD
    U, S, VT = svd(H0, full_matrices=False)
    
    # Truncate to system order
    U_r = U[:, :system_order]
    S_r = np.diag(S[:system_order])
    V_r = VT.T[:, :system_order]
    
    sqrt_inv_S_r = fractional_matrix_power(S_r, -0.5)

    # Compute system matrices
    A = sqrt_inv_S_r @ U_r.T @ H1 @ V_r @ sqrt_inv_S_r
    
    # Compute the reduced-order B matrix
    B = sqrt_inv_S_r @ U_r.T @ H0[:, :num_in]
    
    # Compute the reduced-order C matrix
    C = H0[:num_out, :] @ V_r @ sqrt_inv_S_r
    
    return A, B, C

def APSM_algorithm(combined_matrix, learning_rate, iterations, A, B, C_matrix, P, Q, R, S):
    """
    APSM (Adaptive Physics-Informed System Modeling) algorithm implementation.

    Parameters:
    - combined_matrix (numpy.ndarray): Combined input and output data, where each column represents a time step.
    - learning_rate (float): Learning rate for gradient descent updates to the state transition matrix A.
    - iterations (int): Number of iterations to run the algorithm.
    - A (numpy.ndarray): Initial state transition matrix.
    - B (numpy.ndarray): Control input matrix.
    - C_matrix (numpy.ndarray): Observation matrix.
    - P (numpy.ndarray): Initial error covariance matrix.
    - Q (numpy.ndarray): Process noise covariance matrix.
    - R (numpy.ndarray): Measurement noise covariance matrix.
    - S (numpy.ndarray): Cholesky factor of the process noise covariance (unused in this implementation but included as a parameter).
    - u (numpy.ndarray): Control input (set to zero in this implementation).

    Returns:
    - x_estimates (numpy.ndarray): Estimated states over all iterations.
    - A (numpy.ndarray): Updated state transition matrix after all iterations.
    - y_estimates (numpy.ndarray): Estimated measurements over all iterations.
    - P (numpy.ndarray): Final error covariance matrix.
    """
    # Extract the dimensions of the state (n) and observation (m)
    n = A.shape[0]
    m = C_matrix.shape[0]
    
    # Initialize the state estimate and arrays to store results
    x_hat = np.zeros((n, 1))  # Initial state estimate
    x_estimates = np.zeros((n, iterations))  # To store state estimates for each iteration
    y_estimates = np.zeros((m, iterations))  # To store measurement estimates for each iteration
    
    for k in range(iterations):
        # Save the previous state estimate
        x_hat_old = x_hat
        
        # Extract the current observation vector from the combined matrix
        y_k = combined_matrix[:, k].reshape(-1, 1)
        
        # Initialize the control input to zero (u_k)
        u_k = np.zeros((1, 1))
        
        # Perform Kalman filter update using the current A matrix
        x_hat, P, y_pred = kalman_filter_update(A, B, C_matrix, Q, R, x_hat_old, P, u_k, y_k)
        
        # Update the A matrix using gradient descent
        A = gradient_descent_update(A, x_hat_old, x_hat, learning_rate)
        
        # Store the updated state and measurement estimates
        x_estimates[:, k] = x_hat.flatten()
        y_estimates[:, k] = y_pred.flatten()
    
    return x_estimates, A, y_estimates, P


def load_filtered_data(file_path):
    """
    Load filtered data from a .mat file and convert it to a NumPy array.

    Parameters:
    - file_path (str): Path to the .mat file.

    Returns:
    - np.ndarray: Combined filtered data, arranged as a NumPy array with columns in sequence.
    """
    # Load the .mat file
    mat_data = loadmat(file_path)

    # Filter out metadata from the MATLAB file such as '__header__', '__version__', '__globals__'
    filtered_data_dict = {key: value for key, value in mat_data.items() if not key.startswith('__')}

    # Extract and combine data into a NumPy array
    # Sort keys to ensure data is extracted in order
    sorted_keys = sorted(filtered_data_dict.keys(), key=lambda x: int(x.split('_')[-1]))
    
    # Combine data columns into a single array
    all_data = np.column_stack([filtered_data_dict[key].flatten() for key in sorted_keys])

    return all_data


# Example usage
file_path = 'F:\\博士课题\\小论文\\杭州湾项目论文\\杭州湾大桥数字孪生模型/data/2015-11-01 00-01-filtered_data.mat'

# Load all filtered data
y = load_filtered_data(file_path)

# ============ Parameters ============
# Create a time vector for visualization or analysis
time = np.linspace(0, 10, y.shape[0])

# # Example parameters
# rows = 3000  # Number of rows for the Hankel matrix
# cols = 3000  # Number of columns for the Hankel matrix
# num_in = 1  # Number of input signals
# num_out = 12  # Number of output signals
# system_order = 100  # Order of the system for identification
# sampling_frequency = 50  # Sampling frequency (in Hz)

# # Create and analyze the Hankel matrix
# H = create_hankel_matrix(y, rows + 1, cols)
# compute_and_plot_singular_values_scatter(H)

# # Perform ERA (Eigensystem Realization Algorithm) analysis
# A_era, B_era, C_era = era_analysis(y, rows, cols, num_in, num_out, system_order)

# # Save the computed matrices to local files
# np.save('A_era.npy', A_era)
# np.save('B_era.npy', B_era)
# np.save('C_era.npy', C_era)
# print("Matrices saved.")

# ================= Adaptive Filtering Setup =================
time = np.linspace(0, 3600, y.shape[0])  # Updated time vector for extended time range
n = 100  # State dimension
m = 12  # Observation dimension

# Initialize parameters for the Kalman filter
A = np.eye(n)  # Initial state transition matrix
B = np.zeros((n, 1))  # Control input matrix
P = np.eye(n) * 0.01  # Initial covariance matrix with small values
Q = np.eye(n) * 0.01  # Process noise covariance matrix
R = np.eye(m) * 0.00001  # Measurement noise covariance matrix
S = np.linalg.cholesky(np.eye(n) * 1e-3)  # Cholesky factor of process noise covariance
learning_rate = 0.0002  # Learning rate for gradient descent
iterations = y.shape[0]  # Total number of iterations
iterations_count = 1  # Number of iterations for a specific process

# Initialize a list to store NMSE (Normalized Mean Squared Error) values for each iteration
y_nmse_tests = []
# If running repeatedly, matrices can be loaded from previous runs to continue from the last state
# Uncomment these lines if resuming from saved matrices
# A = np.load('A_update.npy')
# P = np.load('P_update.npy')

# Load initial matrices from ERA results
A = np.load('A_era.npy')
C_matrix = np.load('C_era.npy')

# Record the total start time for the process
total_start_time = tm.time()

# Run the algorithm for a specified number of iterations
for iteration in range(iterations_count):
    # After the first iteration, load updated matrices and NMSE values from previous runs
    if iteration > 0:
        A = np.load('A_update.npy')  # Load updated A matrix
        P = np.load('P_update.npy')  # Load updated P matrix
        y_nmse_tests = np.load('y_nmse_tests.npy').tolist()  # Load NMSE values as a list
    else:
        # Initialize NMSE tracking list for the first iteration
        y_nmse_tests = []

    # Execute the APSM algorithm
    x_estimates_test, A_update, y_estimates_test, P_update = APSM_algorithm(
        y.T, learning_rate, iterations, A, B, C_matrix, P, Q, R, S
    )

    # Save updated matrices for use in subsequent iterations
    np.save('A_update.npy', A_update)
    np.save('P_update.npy', P_update)
    print(f"Matrices saved for iteration {iteration + 1}.")
    
    # Evaluate predictions using MSE, NMSE, and R^2 metrics
    y_mse_test, y_nmse_test, y_r2_test = evaluate_predictions(y.T, y_estimates_test)
    y_nmse_tests.append(y_nmse_test)  # Append the current NMSE to the list
    
    # Save NMSE values to a file for persistence
    np.save('y_nmse_tests.npy', y_nmse_tests)

    # Visualize results and monitor progress
    visualize_results_monit(time, y.T, y_estimates_test, 12, iterations)  # Plot monitored results
    compare_psd(y, y_estimates_test.T, fs=50.0)  # Compare Power Spectral Density (PSD)
    
    # Clear variables to free memory
    del A, P, A_update, P_update, x_estimates_test

# Record the total end time and compute the duration
total_end_time = tm.time()
total_duration = total_end_time - total_start_time
print(f"Total duration for {iterations_count} iterations: {total_duration:.2f} seconds.")

# Save final NMSE values to a file
np.save('y_nmse_tests.npy', y_nmse_tests)

# Reload NMSE values from the saved file for plotting
y_nmse_tests = np.load('y_nmse_tests.npy')

# Plot NMSE values over iterations
plot_y_nmse_over_iterations(y_nmse_tests)

# Print the final NMSE value from the last iteration
print("Final y_nmse_test value:", y_nmse_tests[-1])
print("Learning_rate:", learning_rate)

# Visualize the final results and compare PSDs
visualize_results_monit(time, y.T, y_estimates_test, 12, iterations)
compare_psd(y, y_estimates_test.T, fs=50.0)























