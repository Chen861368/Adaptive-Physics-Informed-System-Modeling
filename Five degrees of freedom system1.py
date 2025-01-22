# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
np.random.seed(1)

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
    
    return x_hat, P, C @ x_hat_pred



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

def visualize_results(time, actual_data, estimated_data, n, iterations):
    """
    Visualize the true and estimated states with stylistic adjustments for enhanced readability
    and aesthetics.

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

        # Plot actual data with a solid black line
        plt.plot(time[:iterations], actual_data[i, :iterations], label=f'Monitoring data', linestyle='-', marker='', color='blue', linewidth=2.5)

        # Plot estimated data with a dashed red line
        plt.plot(time[:iterations], estimated_data[i, :iterations], label=f'Predicted data', linestyle=':', marker='', color='orange', linewidth=2.5)

        # Customize labels with explicit font sizes
        plt.xlabel('Time (seconds)', fontsize=25)
        plt.ylabel('Amplitude', fontsize=25)

        # Enlarge legend line and text, add a black edge to the legend with a white background for visibility
        legend = plt.legend(fontsize='x-large', handlelength=2, edgecolor='black', frameon=True, fancybox=False)
        # Set the linewidth of the legend border
        legend.get_frame().set_linewidth(1.5)

        # Explicitly set tick label sizes
        plt.tick_params(axis='both', which='major', labelsize=20)

        # Use dashed grid lines for better readability
        plt.grid(True, linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.show()


def plot_displacements(time, displacements):
    """
    Plots the displacement of each mass over time with stylistic adjustments
    for enhanced readability and aesthetics.
    """
    sns.set(style="white", context="talk")  # Use a white background and context suitable for presentations

    plt.figure(figsize=(12, 8), dpi=300)  # Set figure size and DPI
    colors = sns.color_palette("muted", displacements.shape[0])  # Use a muted color palette

    # Set global font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16

    for i, color in zip(range(displacements.shape[0]), colors):
        plt.plot(time, displacements[i, :], label=f'Mass {i+1}', color=color)

    # Customize labels with explicit font sizes
    plt.xlabel('Time (seconds)', fontsize=25)
    plt.ylabel('Displacement', fontsize=25)

    # Enlarge legend line and text
    plt.legend(fontsize='x-large', handlelength=2)

    # Explicitly set tick label sizes
    plt.tick_params(axis='both', which='major', labelsize=20)

    # Use dashed grid lines
    plt.grid(True, linestyle='--')

    plt.tight_layout()
    plt.show()



def APSM_algorithm(F_initial, combined_matrix, learning_rate, iterations, A, B, C_matrix, P, Q, R, S):
    """
    Adaptive Projection Subspace Method (APSM) algorithm for system state estimation 
    and adaptive update of the system matrix A using gradient descent.

    Parameters:
    - F_initial (np.ndarray): Initial input force matrix.
    - combined_matrix (np.ndarray): Combined system state matrix, where the first half represents states, 
                                     and the second half represents derivatives.
    - learning_rate (float): Learning rate for gradient descent updates of matrix A.
    - iterations (int): Number of iterations for the algorithm.
    - A (np.ndarray): Initial state transition matrix.
    - B (np.ndarray): Control input matrix.
    - C_matrix (np.ndarray): Observation matrix.
    - P (np.ndarray): Initial error covariance matrix for Kalman filter.
    - Q (np.ndarray): Process noise covariance matrix.
    - R (np.ndarray): Measurement noise covariance matrix.
    - S (np.ndarray): Optional projection matrix (not used in this implementation).

    Returns:
    - x_estimates (np.ndarray): Estimated system states over all iterations (2n x iterations).
    - A (np.ndarray): Adaptively updated state transition matrix.
    - y_estimates (np.ndarray): Predicted observation matrix over all iterations (m x iterations).
    """
    n = combined_matrix.shape[0] // 2  # Number of states
    x_hat = np.zeros((2 * n, 1))  # Initial state estimate
    x_estimates = np.zeros((2 * n, iterations))  # Store estimated states
    y_estimates = np.zeros((C_matrix.shape[0], iterations))  # Store predicted observations

    for k in range(iterations):
        x_hat_old = x_hat
        y_k = C_matrix @ combined_matrix[:, k].reshape(-1, 1)  # Observation at current time step
        u_k = F_initial[:, k].reshape(-1, 1) * 0  # Control input (assumed zero for simplicity)

        # Apply Kalman filter update using the current A matrix
        x_hat, P, y_pred = kalman_filter_update(A, B, C_matrix, Q, R, x_hat_old, P, u_k, y_k)

        # Update the A matrix using gradient descent 
        A = gradient_descent_update(A, x_hat_old, x_hat, learning_rate)
        
        #Update the A matrix using proximal gradient descent 
        # A = gradient_descent_update_diag(A, x_hat_old, x_hat, learning_rate)

        # Save the updated state estimate and predicted observation
        x_estimates[:, k] = x_hat.flatten()
        y_estimates[:, k] = y_pred.flatten()

    return x_estimates, A, y_estimates


def kf_algorithm(F_initial, combined_matrix, iterations, A, B, C_matrix, P, Q, R, S):
    """
    Kalman filter (KF) algorithm for system state estimation without adaptive updates.

    Parameters:
    - F_initial (np.ndarray): Initial input force matrix.
    - combined_matrix (np.ndarray): Combined system state matrix, where the first half represents states, 
                                     and the second half represents derivatives.
    - iterations (int): Number of iterations for the algorithm.
    - A (np.ndarray): State transition matrix.
    - B (np.ndarray): Control input matrix.
    - C_matrix (np.ndarray): Observation matrix.
    - P (np.ndarray): Initial error covariance matrix for Kalman filter.
    - Q (np.ndarray): Process noise covariance matrix.
    - R (np.ndarray): Measurement noise covariance matrix.
    - S (np.ndarray): Optional projection matrix (not used in this implementation).

    Returns:
    - x_estimates (np.ndarray): Estimated system states over all iterations (2n x iterations).
    - y_estimates (np.ndarray): Predicted observation matrix over all iterations (m x iterations).
    """
    n = combined_matrix.shape[0] // 2  # Number of states
    x_hat = np.zeros((2 * n, 1))  # Initial state estimate
    x_estimates = np.zeros((2 * n, iterations))  # Store estimated states
    y_estimates = np.zeros((C_matrix.shape[0], iterations))  # Store predicted observations

    for k in range(iterations):
        x_hat_old = x_hat
        y_k = C_matrix @ combined_matrix[:, k].reshape(-1, 1)  # Observation at current time step
        u_k = F_initial[:, k].reshape(-1, 1) * 0  # Control input (assumed zero for simplicity)

        # Apply Kalman filter update using the current A matrix
        x_hat, P, y_pred = kalman_filter_update(A, B, C_matrix, Q, R, x_hat_old, P, u_k, y_k)

        # Save the updated state estimate and predicted observation
        x_estimates[:, k] = x_hat.flatten()
        y_estimates[:, k] = y_pred.flatten()

    return x_estimates, y_estimates


def plot_frequency_spectrum(time, displacements):
    """
    Plots the frequency spectrum of each mass's displacement using FFT,
    with stylistic adjustments for enhanced readability and aesthetics.
    """
    sns.set(style="white", context="talk")  # Use a white background and context suitable for presentations

    plt.figure(figsize=(12, 8), dpi=300)  # Set figure size and DPI

    # Calculate FFT and frequencies
    dt = time[1] - time[0]  # Calculate timestep
    n = displacements.shape[1]
    freq = np.fft.fftfreq(n, d=dt)[:n//2]  # Only positive frequencies

    colors = sns.color_palette("muted", displacements.shape[0])  # Use a muted color palette

    # Set global font to Times New Roman and increase size
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16

    for i, color in zip(range(displacements.shape[0]), colors):
        fft_vals = np.fft.fft(displacements[i, :])
        fft_theo = 2.0/n * np.abs(fft_vals[:n//2])  # Single-sided spectrum

        plt.plot(freq, fft_theo, label=f'Mass {i+1}', color=color)

    # Customize labels with explicit font sizes
    plt.xlabel('Frequency (Hz)', fontsize=25)
    plt.ylabel('Amplitude', fontsize=25)

    plt.xlim(0, 0.4)  # Adjust frequency display range as needed

    # Enlarge legend line and text
    plt.legend(fontsize='x-large', handlelength=2)

    # Explicitly set tick label sizes
    plt.tick_params(axis='both', which='major', labelsize=20)

    # Use dashed grid lines
    plt.grid(True, linestyle='--')

    plt.tight_layout()
    plt.show()


def visualize_matrix(A, title):
    """
    Visualizes a matrix with a heatmap-style plot.

    Parameters:
    - A: The matrix to be visualized
    - title: Title of the plot
    """
    vmax = np.abs(A).max()
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    X, Y = np.meshgrid(np.arange(A.shape[1]+1), np.arange(A.shape[0]+1))
    ax.invert_yaxis()
    pos = ax.pcolormesh(X, Y, A.real, cmap="seismic", vmax=vmax, vmin=-vmax)
    cbar = fig.colorbar(pos, ax=ax)
    cbar.ax.tick_params(labelsize=10)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.show()  # Display the figure



# ============ Parameters ============

c = 0.05  # Damping coefficient
k = 1  # Stiffness coefficient
m = 6  # Number of observable degrees of freedom
n = 5  # Total number of degrees of freedom
t_end = 1000  # Simulation end time
dt = 0.005  # Time step size
time = np.arange(0, t_end + dt, dt)  # Time array
noise_ratio = 0.1  # Noise ratio for added non-stationary Gaussian noise
learning_rate = 100  # Learning rate for the adaptive algorithm
iterations = len(time)  # Number of iterations for computation and plotting
rank = 10  # Rank used for the DMD matrix

# Mass matrix M
M = 1.5 * np.eye(n)  # Diagonal mass matrix

# Stiffness matrix K
# Constructed as a tridiagonal matrix
K = np.diag([2 * k] * (n - 1) + [k]) - np.diag([k] * (n - 1), 1) - np.diag([k] * (n - 1), -1)

# Damping matrix C
C = (c / k) * K  # Proportional damping matrix

# External force F(t), applied only at the first time step to the last mass
F_initial = np.zeros((n, len(time)))
F_initial[-1, 0] = np.random.normal(0, 0.5) * 2  # Random force applied at the first step

# Initial conditions
x0 = np.zeros(n)  # Initial displacements
v0 = np.zeros(n)  # Initial velocities
# Initial accelerations, computed using the equation of motion
a0 = np.linalg.inv(M).dot(F_initial[:, 0] - C.dot(v0) - K.dot(x0))

# Generate data using the Newmark-beta method
displacements, velocities, accelerations = newmark_beta_modified(dt, K, M, C, F_initial, x0, v0, a0)

# Add non-stationary Gaussian noise to the data
noisy_displacements = add_nonstationary_gaussian_noise(displacements, noise_ratio)
noisy_velocities = add_nonstationary_gaussian_noise(velocities, noise_ratio)

# Combine displacements and velocities into a single matrix
combined_matrix = np.vstack((displacements, velocities))
combined_matrix_noise = np.vstack((noisy_displacements, noisy_velocities))

# Initialize parameters for the adaptive and Kalman filter algorithms
A = compute_dmd_matrix(combined_matrix, rank)  # Exact initial state transition matrix (DMD)

B = np.random.randn(2 * n, n)  # Control input matrix with random values
# Add perturbations to matrices A and B
A_perturbed, B_perturbed = perturb_matrices(A, B, noise_level=0.9)

C_matrix = np.random.randn(m, 2 * n)  # Output (observation) matrix, assuming incomplete observations

P = np.eye(2 * n) * 1e-1  # Initial covariance matrix with small values
Q = np.eye(2 * n) * 1e-1  # Process noise covariance matrix
R = np.eye(m) * 2e-1  # Measurement noise covariance matrix
S = np.linalg.cholesky(np.eye(2 * n) * 1e-3)  # Cholesky decomposition for optional usage


# Call the APSM algorithm for adaptive state estimation and matrix update
x_estimates_opi, A_updated, y_estimates_opi = APSM_algorithm(
    F_initial, combined_matrix_noise, learning_rate, iterations, 
    A_perturbed, B, C_matrix, P, Q, R, S
)

# Call the Kalman filter (KF) algorithm for state estimation without adaptive updates
x_estimates_kf, y_estimates_kf = kf_algorithm(
    F_initial, combined_matrix_noise, iterations, 
    A_perturbed, B, C_matrix, P, Q, R, S
)

# Slice `combined_matrix` and estimates to analyze specific parts of the data
slice_number = 0  # Starting index for slicing
combined_matrix_sliced = combined_matrix[:, slice_number:]  # Sliced actual data
x_estimates_opi_sliced = x_estimates_opi[:, slice_number:]  # Sliced APSM estimates
y_estimates_opi_sliced = y_estimates_opi[:, slice_number:]  # Sliced APSM predictions
x_estimates_kf_sliced = x_estimates_kf[:, slice_number:]  # Sliced KF estimates
time_sliced = time[slice_number:]  # Sliced time vector
iterations_sliced = iterations - slice_number  # Adjusted number of iterations

# Compare the estimated `x` values with the actual `x` values
x_mse_opi, x_nmse_opi, x_r2_opi = evaluate_predictions(combined_matrix, x_estimates_opi)  # Metrics for APSM
x_mse_kf, x_nmse_kf, x_r2_kf = evaluate_predictions(combined_matrix, x_estimates_kf)  # Metrics for KF

# Compare the estimated `y` values with the actual `y` values
y_es_mse_opi, y_es_nmse_opi, y_es_r2_opi = evaluate_predictions(
    C_matrix @ combined_matrix_noise, y_estimates_opi
)  # Metrics for APSM
y_mse_kf, y_nmse_kf, y_r2_kf = evaluate_predictions(
    C_matrix @ combined_matrix_noise, y_estimates_kf
)  # Metrics for KF

# Print evaluation results for `y` estimates
print(f"y estimates APSM Normalized Mean Squared Error (NMSE): {y_es_nmse_opi}")
print(f"y estimates KF Normalized Mean Squared Error (NMSE): {y_nmse_kf}")

# Calculate and display the Frobenius norm difference between the true `A` matrix and the updated matrix
frobenius_diff_updated = np.linalg.norm(A - A_updated, 'fro')
print(f"frobenius_diff_updated: {frobenius_diff_updated}")

# Plot displacement and acceleration frequency spectra
plot_displacements(time, displacements)  # Plot displacements over time
plot_frequency_spectrum(time, accelerations)  # Plot frequency spectrum of accelerations

# Visualize the noisy vs. noise-free data comparison
visualize_results(
    time_sliced, combined_matrix_noise, combined_matrix, 2 * n, iterations_sliced
)

# Visualize matrices: true, updated, and perturbed state transition matrices
visualize_matrix(A, "A")  # Original A matrix
visualize_matrix(A_updated, "A_updated")  # Updated A matrix
visualize_matrix(A_perturbed, "A_perturbed")  # Perturbed A matrix

# Visualize comparison of `y` estimates vs. monitored `y` values for APSM
visualize_results(
    time_sliced, C_matrix @ combined_matrix_noise, y_estimates_opi, m, iterations_sliced
)



