# Adaptive Physics-Informed System Modeling (APSM)

Code for **"Adaptive Physics-Informed System Modeling (APSM) for Online Structural Dynamic Simulation"**. This repository implements the APSM algorithm, designed to overcome challenges in modeling nonlinear dynamics, noise, and data sparsity by leveraging real-time monitoring data and integrating residual-based updates with fundamental physical principles.

---

## Abstract

To track the lifetime performance of structural systems, time-varying models are highly effective as they can accurately capture the system's dynamic and evolving states. However, existing model updating methods face challenges in efficiency and adaptability, making real-time simulation of structural systems difficult.

This repository implements **Adaptive Physics-Informed System Modeling (APSM)**, a novel approach specifically designed for:
- Online identification of structural conditions.
- Accurate simulation of structural dynamic responses.

Key features of APSM:
- Employs time-varying state-space models to address nonlinear dynamics.
- Integrates residual-based updates with physical information at each time step.
- Produces system matrices with clear physical significance, offering insights into dynamic properties.

The proposed method has been validated using:
- A five-degree-of-freedom numerical case study.
- Real-world monitoring data from the Hangzhou Bay Bridge.

### Key Results:
- **Robustness and High Precision:** Achieved a Normalized Mean Squared Error (NMSE) of **2.107%** for time-domain predictions of main beam acceleration data from the actual engineering case of the Hangzhou Bay Bridge, significantly outperforming classical methods.
- **Efficiency:** Processes **1 hour of monitoring data in 33 seconds**.
- **Generalization Performance:** The model retains superior accuracy when applied to unseen data after 1 hour of online updating.

---

## Highlights
- **Novel Approach:** Proposed a model updating method using real-time sensor data for online structural dynamic simulation.
- **Low Computational Complexity:** Requires only 33 seconds to process 1 hour of monitoring data.
- **Strong Generalization:** After 1 hour of online updates, the fixed model demonstrates high accuracy in simulating structural dynamics, outperforming classical time-domain methods.

---

## Installation and Usage

### Required Libraries
To run the code, ensure the following Python libraries are installed:

```python
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
```

### Usage

To use the code, replace the default data file path in the code with your own data file path, for example:

```python
file_path = 'F:\\DoctoralResearch\\Paper\\HangzhouBayProject\\HangzhouBayBridgeDigitalTwinModel/data/2015-11-01 00-01-filtered_data.mat'
```

And update the save path to your desired directory, for example:

```python
save_path = "C:\\Users\\HIT\\Desktop" 
```

Ensure that the dimensions of your data match the requirements of the code for correct execution.

### Five Degrees of Freedom System1

This section contains the **five degrees of freedom system** numerical simulation data and implementation code used in the paper for simulation and validation.

### Five Degrees of Freedom System2

This section is used to compare the performance of the **APSM algorithm** under two conditions: **with physical constraints** and **without physical constraints**, in terms of prediction accuracy and system matrix identification.

- **Notes**:
  1. The code has a fixed random seed, so you should be able to reproduce the results in the paper.
  2. **Run the code for the case with physical constraints first, and then run the code for the case without physical constraints separately.**
  3. **Do not run both codes simultaneously**, as differences in random numbers may result in non-comparable performance.
  4. To ensure fair comparisons, run the two cases independently.

### APSM_Algorithm

The `APSM_Algorithm` file contains the **ERA (Eigensystem Realization Algorithm)** and the implementation code for processing the **Hangzhou Bay Bridge main beam acceleration data**.

### Notes

- The provided APSM algorithm code uses ERA (Eigensystem Realization Algorithm) as an example to construct the state-space model.
- In practice, you can use other methods to build the state-space model, such as:
  - **DMD (Dynamic Mode Decomposition)**
  - **SSI (Stochastic Subspace Identification)**
  - **NExT + ERA**, and others.

Make sure to choose the appropriate algorithm and modeling method based on your specific requirements and application scenarios.

