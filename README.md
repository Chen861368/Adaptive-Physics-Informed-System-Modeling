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
To use the code, replace the default data file path in the code:

```python
file_path = 'F:\\博士课题\\小论文\\杭州湾项目论文\\杭州湾大桥数字孪生模型/data/2015-11-01 00-01-filtered_data.mat'
```

with your own data file path. Ensure that the dimensions of your data match the requirements of the code for correct execution.


