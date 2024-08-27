# LPBF Process Analysis and Optimization

This repository contains a collection of Jupyter notebooks for analyzing and optimizing Laser Powder Bed Fusion (LPBF) processes. The notebooks cover various aspects of LPBF, including experimental design, signal analysis, and machine learning approaches for process optimization.

## Repository Structure

The repository consists of the following Jupyter notebooks:

1. `00_DoE.ipynb`: Detailed Design of Experiments (DoE) with a full list of parameters used
2. `00_layer_thickness_evolution.ipynb`: Analysis of layer thickness evolution from a mathematical perspective
3. `01_EDA_316L.ipynb`: Exploratory Data Analysis (EDA) and visualization of raw signals for 316L stainless steel
4. `01_EDA_Ti64.ipynb`: EDA and visualization of raw signals for Ti-6Al-4V
5. `01_Plot_raw_signals.ipynb`: Code to display the raw data
6. `02_Segmentation_316L.ipynb`: Demonstration of signal segmentation for 316L
7. `02_Segmentation_Ti64.ipynb`: Demonstration of signal segmentation for Ti-6Al-4V
8. `03_Ground_truth inspection.ipynb`: Analysis of melting regime detection results from cross-sections
9. `04_Plot_features.ipynb`: Analysis of statistical features extracted from optical data vs. scanning speed
10. `05_Clustering.ipynb`: Performance analysis of clustering algorithms and comparison of multiple algorithms
11. `06_Calculate length scale for GPC.ipynb`: Initialization of length scale for Gaussian Process Classification (GPC)
12. `06_Iterative sampling-multiple_runs.ipynb`: Implementation and analysis of an uncertainty-driven algorithm for determining processing maps

## Notebook Descriptions

### Experimental Design and Theoretical Analysis
- `00_DoE.ipynb`: Contains the detailed Design of Experiments, including all parameters used in the LPBF process.
- `00_layer_thickness_evolution.ipynb`: Analyzes the evolution of layer thickness from a mathematical perspective, considering various shrinkage factors and nominal layer thicknesses.

### Data Visualization and Exploratory Analysis
- `01_EDA_316L.ipynb` and `01_EDA_Ti64.ipynb`: Visualize raw signals for 316L stainless steel and Ti-6Al-4V, respectively.
- `01_Plot_raw_signals.ipynb`: Provides code for displaying raw data from LPBF processes.

### Signal Processing
- `02_Segmentation_316L.ipynb` and `02_Segmentation_Ti64.ipynb`: Demonstrate the signal segmentation process for 316L and Ti-6Al-4V materials.

### Ground Truth and Feature Analysis
- `03_Ground_truth inspection.ipynb`: Analyzes the results of melting regime detection from cross-sectional data.
- `04_Plot_features.ipynb`: Examines the relationship between extracted statistical features from optical data and scanning speed.

### Machine Learning Approaches
- `05_Clustering.ipynb`: Evaluates the performance of various clustering algorithms for process parameter classification.
- `06_Calculate length scale for GPC.ipynb`: Demonstrates how to initialize the length scale for Gaussian Process Classification based on input data distribution.
- `06_Iterative sampling-multiple_runs.ipynb`: Implements and analyzes an uncertainty-driven algorithm for determining optimal processing maps.

## Usage

To use these notebooks:

1. Clone this repository to your local machine.
2. Ensure you have Jupyter Notebook or JupyterLab installed, along with Python 3.x.
3. Install the required dependencies (list of dependencies to be added).
4. Open the notebooks in Jupyter and run the cells sequentially.

Note: Some notebooks may require specific datasets, which should be placed in the appropriate directory (details to be provided).

## Dependencies

This project requires the following Python libraries:
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- SciPy

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.


## Contact

For any questions or feedback regarding this project, please contact Giulio Masinelli at giulio.klipsch@gmail.com.
