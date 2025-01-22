# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
np.random.seed(1)

def compute_dmd_matrix(displacements, rank=None):
    """
    Compute the Dynamic Mode Decomposition (DMD) matrix A with an option to use truncated SVD.

    Parameters:
    - displacements: numpy.ndarray, a matrix containing the displacements of each mass at each time step.
    - rank: int, optional, the rank for the truncated SVD.

    Returns:
    - A: numpy.ndarray, the approximated system matrix describing the dynamics.
    """
    # Split the displacement data into X and Y matrices
    X = displacements[:, :-1]
    Y = displacements[:, 1:]

    # Perform the Singular Value Decomposition (SVD) of X
    U, Sigma, Vh = np.linalg.svd(X, full_matrices=False)

    # If a rank is specified, truncate the SVD results
    if rank is not None and rank > 0:
        U = U[:, :rank]
        Sigma = Sigma[:rank]
        Vh = Vh[:rank, :]

    # Construct the diagonal matrix for the pseudo-inverse
    Sigma_inv = np.linalg.inv(np.diag(Sigma))

    # Compute the DMD matrix A
    A = Y @ Vh.T @ Sigma_inv @ U.T

    return A

def project_to_cyclic_matrix(matrix):
    """
    Efficiently project a matrix to a cyclic matrix based on its first row,
    avoiding issues with np.roll for non-scalar shift values.

    Parameters:
    - matrix: A NumPy array representing the original matrix.

    Returns:
    - A NumPy array representing the projected cyclic matrix.
    """
    first_row = matrix[0, :]
    n = first_row.size

    # Initialize the cyclic matrix
    cyclic_matrix = np.zeros_like(matrix)

    # Manually create the cyclic shifts
    for i in range(n):
        cyclic_matrix[i, :] = np.concatenate((first_row[-i:], first_row[:-i]))

    return cyclic_matrix

def gradient_descent_update(A_k, x_k, y_k, learning_rate):
    """
    Performs a gradient descent update for the circulant matrix A_k 
    using a single column vector x_k and its corresponding scalar y_k.

    Parameters:
    - A_k (np.ndarray): Current estimate of the circulant matrix (n x n).
    - x_k (np.ndarray): Input column vector (n x 1).
    - y_k (float): Observed scalar corresponding to x_k.
    - learning_rate (float): Step size for the gradient descent.

    Returns:
    - A_k (np.ndarray): Updated circulant matrix after the gradient descent step.
    """
    # Compute the gradient
    grad = -2 * (y_k - A_k.dot(x_k)) * x_k.T

    # Update A_k using the gradient
    A_k -= learning_rate * grad

    return A_k


def gradient_descent_update_diag(A_k, x_k, y_k, learning_rate):
    """
    Performs a gradient descent update for the circulant matrix A_k 
    using a single column vector x_k and its corresponding scalar y_k. 
    Ensures that A_k remains a circulant matrix by projecting to a lower triangular form.

    Parameters:
    - A_k (np.ndarray): Current estimate of the circulant matrix (n x n).
    - x_k (np.ndarray): Input column vector (n x 1).
    - y_k (float): Observed scalar corresponding to x_k.
    - learning_rate (float): Step size for the gradient descent.

    Returns:
    - A_k (np.ndarray): Updated circulant matrix after the gradient descent step.
    """
    # Compute the gradient
    grad = -2 * (y_k - A_k.dot(x_k)) * x_k.T

    # Update A_k using the gradient
    A_k -= learning_rate * grad

    # Ensure A_k remains a circulant matrix by projecting to a lower triangular matrix
    A_k = np.tril(A_k)

    return project_to_cyclic_matrix(A_k)


def evaluate_predictions(actual, predicted):
    """
    Evaluates the quality of predictions using three metrics: 
    Mean Squared Error (MSE), Normalized Mean Squared Error (NMSE), 
    and Coefficient of Determination (R^2). 

    Handles complex data by considering only the real part of the inputs.

    Parameters:
    - actual (np.ndarray): The actual data, which may contain complex values.
    - predicted (np.ndarray): The predicted data, which may contain complex values.

    Returns:
    - mse (float): Mean Squared Error of the predictions.
    - nmse (float): Normalized Mean Squared Error of the predictions.
    - r2 (float): Coefficient of Determination (R^2) of the predictions.
    """
    # Use only the real part of the data for evaluation
    actual_real = np.real(actual)
    predicted_real = np.real(predicted)
    
    # Compute the Mean Squared Error (MSE)
    mse = mean_squared_error(actual_real, predicted_real)
    
    # Compute the Normalized Mean Squared Error (NMSE)
    variance_actual = np.var(actual_real)
    nmse = mse / variance_actual
    
    # Compute the Coefficient of Determination (R^2)
    r2 = r2_score(actual_real, predicted_real)
    
    return mse, nmse, r2





def add_nonstationary_gaussian_noise(signal, noise_ratio):
    """
    Add non-stationary Gaussian noise to a signal. The noise added to each sample is proportional
    to the magnitude of the signal at that point.

    Parameters:
    - signal (np.ndarray): The original signal.
    - noise_ratio (float): The ratio of the noise amplitude to the signal amplitude.

    Returns:
    - noisy_signal (np.ndarray): Signal with added non-stationary Gaussian noise.
    """
    # Calculate noise standard deviation for each sample
    noise_std_per_sample = np.abs(signal) * noise_ratio

    # Generate non-stationary Gaussian noise
    noise = noise_std_per_sample * np.random.normal(0, 1, signal.shape)

    # Add noise to the original signal
    noisy_signal = signal + noise
    return noisy_signal

def perturb_matrices(A, B, noise_level=0.01):
    """
    Adds small perturbations to matrices A and B.

    Parameters:
    - A (np.ndarray): Original matrix A, with dimensions n x n.
    - B (np.ndarray): Original matrix B, with dimensions n x k.
    - noise_level (float): Intensity of the noise, default is 0.01.

    Returns:
    - np.ndarray: Perturbed matrix A.
    - np.ndarray: Perturbed matrix B.
    """
    n = A.shape[0]
    k = B.shape[1]

    # Add small random noise to matrix A
    A_perturbed = A + noise_level * np.random.randn(n, n)

    # Add small random noise to matrix B
    B_perturbed = B + noise_level * np.random.randn(n, k)

    return A_perturbed, B_perturbed

def newmark_beta_modified(dt, K, M, C, P, u0, v0, a0):
    """
    Solves the dynamic response of a system using the modified Newmark-beta method.

    Parameters:
    - dt (float): Time step size.
    - K (np.ndarray): Stiffness matrix (n x n).
    - M (np.ndarray): Mass matrix (n x n).
    - C (np.ndarray): Damping matrix (n x n).
    - P (np.ndarray): External force matrix (n x nt).
    - u0 (np.ndarray): Initial displacement vector (n,).
    - v0 (np.ndarray): Initial velocity vector (n,).
    - a0 (np.ndarray): Initial acceleration vector (n,).

    Returns:
    - u (np.ndarray): Displacement matrix over time (n x nt).
    - v (np.ndarray): Velocity matrix over time (n x nt).
    - a (np.ndarray): Acceleration matrix over time (n x nt).
    """
    n = K.shape[0]  # Number of degrees of freedom
    nt = P.shape[1]  # Total number of time steps
    gamma = 0.5
    beta = 0.25

    # Precompute coefficients for the Newmark-beta method
    p = [1 / (beta * (dt ** 2)), 
         gamma / (beta * dt),
         1 / (beta * dt),
         0.5 / beta - 1,
         gamma / beta - 1,
         dt * (gamma / (2 * beta) - 1),
         dt * (1 - gamma),
         gamma * dt]

    # Initialize displacement, velocity, and acceleration arrays
    u = np.zeros((n, nt))
    v = np.zeros((n, nt))
    a = np.zeros((n, nt))

    # Initial conditions
    u[:, 0] = u0
    v[:, 0] = v0
    a[:, 0] = a0

    # Effective stiffness matrix
    K_ = K + p[0] * M + p[1] * C
    K_inv = np.linalg.inv(K_)

    # Time-stepping solution
    for i in range(1, nt):
        # Apply external force only during the first time step
        if i == 1:
            P_ = P[:, 0] + M.dot(p[0] * u[:, i-1] + p[2] * v[:, i-1] + p[3] * a[:, i-1]) + C.dot(p[1] * u[:, i-1] + p[4] * v[:, i-1] + p[5] * a[:, i-1])
        else:
            P_ = M.dot(p[0] * u[:, i-1] + p[2] * v[:, i-1] + p[3] * a[:, i-1]) + C.dot(p[1] * u[:, i-1] + p[4] * v[:, i-1] + p[5] * a[:, i-1])
        
        # Solve for displacement at the current time step
        u[:, i] = K_inv.dot(P_)

        # Calculate acceleration and velocity at the current time step
        a[:, i] = p[0] * (u[:, i] - u[:, i-1]) - p[2] * v[:, i-1] - p[3] * a[:, i-1]
        v[:, i] = v[:, i-1] + p[6] * a[:, i-1] + p[7] * a[:, i]
        
    return u, v, a
def kalman_filter_update(A, B, C, Q, R, x_hat, P, u, y):
    def predict(x_hat, P, A, B, u, Q):
        x_hat_pred = A @ x_hat + B @ u
        P_pred = A @ P @ A.T + Q
        return x_hat_pred, P_pred

    def update(x_hat_pred, P_pred, y, C, R):
        S = C @ P_pred @ C.T + R
        K = P_pred @ C.T @ np.linalg.inv(S)
        x_hat = x_hat_pred + K @ (y - C @ x_hat_pred)
        P = (np.eye(len(P_pred)) - K @ C) @ P_pred
        return x_hat, P
    x_hat_pred, P_pred = predict(x_hat, P, A, B, u, Q)
    x_hat, P = update(x_hat_pred, P_pred, y, C, R)
    return x_hat, P, x_hat_pred


def APSM_con_algorithm(F_initial, combined_matrix, learning_rate, iterations, A, B, C_matrix, P, Q, R, S):
    n = combined_matrix.shape[0] // 2
    x_hat = np.zeros((2*n, 1))
    x_estimates = np.zeros((2*n, iterations))
    y_estimates = np.zeros((2*n, iterations))
    for k in range(iterations):
        x_hat_old = x_hat
        y_k = C_matrix @ combined_matrix[:, k].reshape(-1, 1)
        u_k = F_initial[:, k].reshape(-1, 1) * 0
        x_hat, P, x_pred = kalman_filter_update(A, B, C_matrix, Q, R, x_hat_old, P, u_k, y_k)
        A = gradient_descent_update_diag(A, x_hat_old, x_hat, learning_rate)
        x_estimates[:, k] = x_hat.flatten()
        y_estimates[:, k] = x_pred.flatten()
    return x_estimates, A, y_estimates



def APSM_uncon_algorithm(F_initial, combined_matrix, learning_rate, iterations, A, B, C_matrix, P, Q, R, S):
    n = combined_matrix.shape[0] // 2
    x_hat = np.zeros((2*n, 1))
    x_estimates = np.zeros((2*n, iterations))
    y_estimates = np.zeros((2*n, iterations))
    for k in range(iterations):
        x_hat_old = x_hat
        y_k = C_matrix @ combined_matrix[:, k].reshape(-1, 1)
        u_k = F_initial[:, k].reshape(-1, 1) * 0
        x_hat, P, x_pred = kalman_filter_update(A, B, C_matrix, Q, R, x_hat_old, P, u_k, y_k)
        A = gradient_descent_update(A, x_hat_old, x_hat, learning_rate)
        x_estimates[:, k] = x_hat.flatten()
        y_estimates[:, k] = x_pred.flatten()
    return x_estimates, A, y_estimates


def kf_algorithm(F_initial, combined_matrix, iterations, A, B, C_matrix, P, Q, R, S):
    n = combined_matrix.shape[0] // 2
    x_hat = np.zeros((2*n, 1))
    x_estimates = np.zeros((2*n, iterations))
    y_estimates = np.zeros((2*n, iterations))

    for k in range(iterations):
        x_hat_old = x_hat
        y_k = C_matrix @ combined_matrix[:, k].reshape(-1, 1)
        u_k = F_initial[:, k].reshape(-1, 1)*0
        x_hat, P, x_pred = kalman_filter_update(A, B, C_matrix, Q, R, x_hat_old, P, u_k, y_k)
        x_estimates[:, k] = x_hat.flatten()
        y_estimates[:, k] = x_pred.flatten()
        
    return x_estimates, y_estimates

# ============ Parameters ============

c = 0.05  # Damping coefficient
k = 1  # Stiffness coefficient
n = 5  # Number of degrees of freedom
t_end = 1000  # End time for simulation
dt = 0.005  # Time step size
time = np.arange(0, t_end + dt, dt)  # Time array
noise_ratio = 0.1  # Noise ratio for added non-stationary Gaussian noise
learning_rate = 1300  # Learning rate for adaptive algorithms
iterations = len(time)  # Number of iterations for computation and plotting
rank = 10  # Rank used for the Dynamic Mode Decomposition (DMD)

# Mass matrix M (diagonal matrix where each degree of freedom has the same mass)
M = 1.5 * np.eye(n)

# Stiffness matrix K (tridiagonal matrix representing spring connections)
K = (
    np.diag([2 * k] * (n - 1) + [k])  # Diagonal elements
    - np.diag([k] * (n - 1), 1)      # Sub-diagonal elements
    - np.diag([k] * (n - 1), -1)     # Super-diagonal elements
)

# Damping matrix C (proportional to the stiffness matrix)
C = (c / k) * K

# External force F(t), applied only at the first time step to the last mass
F_initial = np.zeros((n, len(time)))  # Initialize force matrix
F_initial[-1, 0] = np.random.normal(0, 0.5) * 2  # Apply random force at the first step

# Initial conditions for the system
x0 = np.zeros(n)  # Initial displacements (all zeros)
v0 = np.zeros(n)  # Initial velocities (all zeros)
# Initial accelerations, computed from the equation of motion
a0 = np.linalg.inv(M).dot(F_initial[:, 0] - C.dot(v0) - K.dot(x0))

# Generate system response data using the Newmark-beta method
displacements, velocities, accelerations = newmark_beta_modified(dt, K, M, C, F_initial, x0, v0, a0)

# Add non-stationary Gaussian noise to displacements and velocities
noisy_displacements = add_nonstationary_gaussian_noise(displacements, noise_ratio)
noisy_velocities = add_nonstationary_gaussian_noise(velocities, noise_ratio)

# Combine displacements and velocities into a single matrix
combined_matrix = np.vstack((displacements, velocities))  # Noise-free data
combined_matrix_noise = np.vstack((noisy_displacements, noisy_velocities))  # Noisy data

# Initialize parameters for adaptive and state estimation algorithms
A = compute_dmd_matrix(combined_matrix, rank)  # Accurate initial state transition matrix (DMD)
B = np.random.randn(2 * n, n)  # Control input matrix with random values
# Perturb state transition matrix A and control input matrix B to simulate imperfect initial estimates
A_perturbed, B_perturbed = perturb_matrices(A, B, noise_level=0.9)

# Simulate the system for different values of m (number of observable outputs)
m_values = [10, 8, 6, 4, 2]  # List of different observable output dimensions
results = []  # List to store results for analysis
save_path = "C:\\Users\\HIT\\Desktop"  # Change this to your own save path


# Results with physical constraints
for m in m_values:
    print(m)
    # Generate output (observation) matrix, assuming incomplete observations
    C_matrix = np.random.randn(m, 2 * n)
    # Initialize covariance matrices for Kalman filtering
    P = np.eye(2 * n) * 1e-1  # Initial error covariance matrix
    Q = np.eye(2 * n) * 1e-1  # Process noise covariance
    R = np.eye(m) * 2e-1  # Measurement noise covariance
    S = np.linalg.cholesky(np.eye(2 * n) * 1e-3)  # Cholesky decomposition for numerical stability

    # Call the APSM algorithm with physical constraints
    x_estimates_opi, A_updated, y_estimates_opi = APSM_con_algorithm(
        F_initial, combined_matrix_noise, learning_rate, iterations, 
        A_perturbed, B, C_matrix, P, Q, R, S
    )

    # Compare the estimated `x` values with the actual `x` values
    x_mse_opi, x_nmse_opi, x_r2_opi = evaluate_predictions(combined_matrix, y_estimates_opi)

    # Compare the estimated `y` values with the actual `y` values
    y_es_mse_opi, y_es_nmse_opi, y_es_r2_opi = evaluate_predictions(
        C_matrix @ combined_matrix_noise, C_matrix @ y_estimates_opi
    )

    # Calculate the Frobenius norm difference between the true `A` matrix and the updated `A` matrix
    frobenius_diff_updated = np.linalg.norm(A - A_updated, 'fro')

    # Append results to the results list for later analysis
    results.append({
        'm': m,  # Number of outputs (observation dimensions)
        'x_nmse_opi': x_nmse_opi,  # Normalized Mean Squared Error for `x` estimates
        'y_es_nmse_opi': y_es_nmse_opi,  # Normalized Mean Squared Error for `y` estimates
        'frobenius_diff_updated': frobenius_diff_updated  # Frobenius norm difference
    })

# Save results to a CSV file (for physical constraint results)
df_results = pd.DataFrame(results)
df_results.to_csv(f"{save_path}\\simulation_opidmd_results.csv", index=False)
print("Results saved to simulation_results.csv")


# # Results without physical constraints
# for m in m_values:
#     print(m)
#     # Generate output (observation) matrix, assuming incomplete observations
#     C_matrix = np.random.randn(m, 2 * n)
#     # Initialize covariance matrices for Kalman filtering
#     P = np.eye(2 * n) * 1e-1  # Initial error covariance matrix
#     Q = np.eye(2 * n) * 1e-1  # Process noise covariance
#     R = np.eye(m) * 2e-1  # Measurement noise covariance
#     S = np.linalg.cholesky(np.eye(2 * n) * 1e-3)  # Cholesky decomposition for numerical stability

#     # Call the APSM algorithm without physical constraints
#     x_estimates_odmd, A_updated, y_estimates_odmd = APSM_uncon_algorithm(
#         F_initial, combined_matrix_noise, learning_rate, iterations, 
#         A_perturbed, B, C_matrix, P, Q, R, S
#     )

#     # Compare the estimated `x` values with the actual `x` values
#     x_mse_odmd, x_nmse_odmd, x_r2_odmd = evaluate_predictions(combined_matrix, y_estimates_odmd)

#     # Compare the estimated `y` values with the actual `y` values
#     y_es_mse_odmd, y_es_nmse_odmd, y_es_r2_odmd = evaluate_predictions(
#         C_matrix @ combined_matrix_noise, C_matrix @ y_estimates_odmd
#     )

#     # Calculate the Frobenius norm difference between the true `A` matrix and the updated `A` matrix
#     frobenius_diff_updated = np.linalg.norm(A - A_updated, 'fro')

#     # Append results to the results list for later analysis
#     results.append({
#         'm': m,  # Number of outputs (observation dimensions)
#         'x_nmse_odmd': x_nmse_odmd,  # Normalized Mean Squared Error for `x` estimates
#         'y_es_nmse_odmd': y_es_nmse_odmd,  # Normalized Mean Squared Error for `y` estimates
#         'frobenius_diff_updated': frobenius_diff_updated  # Frobenius norm difference
#     })

# # Save results to a CSV file (for results without physical constraints)
# df_results = pd.DataFrame(results)
# df_results.to_csv(f"{save_path}\\simulation_odmd_results.csv", index=False)
# print("Results saved to simulation_results.csv")































