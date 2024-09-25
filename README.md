# NA-MEMD for Python


## Introduction
Python implementation of Noise-Assisted Multivariate Empirical Mode Decomposition (NA-MEMD). This repository contains functions for noise generation, multivariate empirical mode breakdown (MEMD), basic visualization, etc.  

Multivariate Empirical Mode Decomposition (MEMD) is an extension of the traditional Empirical Mode Decomposition (EMD) method, which is used to decompose non-linear and non-stationary signals into simpler oscillatory components known as Intrinsic Mode Functions (IMFs). The key feature of MEMD is its ability to handle multivariate signals, meaning it can simultaneously decompose multiple related signals (or different dimensions of a signal) while ensuring that the decomposition is consistent across all channels.

Noise-Assisted Multivariate Empirical Mode Decomposition (NA-MEMD) is an enhancement of MEMD that incorporates noise into the decomposition process. The noise is added to make the decomposition more robust and to help separate closely spaced frequency components that traditional MEMD might struggle to distinguish. The idea is that the added noise helps to explore the signal more effectively and results in more accurate decomposition.
  
__Primary Benefits of Repository__ : 
- Offers additional options for noise generation from signals
- Allows n-dimensional MEMD as opposed to previous bivariate and quadrivariate implementations
- Implementation is exponentially faster than similar, existing models 
- Basic Visualization tools
- etc.


## Dependencies 
- NumPy
- SciPy  
- sys
- math
- matplotlib

```bash
# sys and math are part of Python Standard Libraries
pip install numpy scipy matplotlib
```


## General Functions and Usage  
```python
signal = np.random.randn(5, 1000)
memd_imfs = memd(signal)  # traditional memd signal processing
na_memd_imfs = na_memd(signal, method="na_fix")  # na_memd processing with noise assistance
add_noise = add_noise(signal, method = "na_fix")  # add noise only to signal without EMD processing
```
For additional functionalities and usage explanations, please reference the example.py file.


## Acknowledgements
Several existing packages and repositories were referenced in the creation of this library. All credit goes to these authors for their contributions to the field.
* https://github.com/AaronLi43/ginkgo_glasgow [1]
* https://github.com/laszukdawid/PyEMD/tree/master
* https://github.com/mariogrune/MEMD-Python-/tree/master [2]
* https://github.com/STherese/NA-MEMD-for-EEG


## Citations
[1] Y. Zhang, G. Wang, Z. Li, et al., "Matlab Open Source Code: Noise-Assisted Multivariate Empirical Mode Decomposition Based Causal Decomposition for Causality Inference of Bivariate Time Series," Front. Neuroinform., vol. 16, Art. no. 851645, Jun. 2022. doi: 10.3389/fninf.2022.851645  [LINK](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9243260/)  
[2] “Research | Empirical Mode Decomposition (EMD), Multivariate EMD, Matlab code and data sources ∴ Dr. Danilo P. Mandic,” www.commsp.ee.ic.ac.uk. https://www.commsp.ee.ic.ac.uk/~mandic/research/emd.htm (accessed Jul. 09, 2024)  [LINK](https://www.commsp.ee.ic.ac.uk/~mandic/research/emd.htm)

## Other
All plans for future updates to the program are listed within the example.py file as comments.

As this is a work in progress, if there are any issues regarding output or code, please email zhue@ufl.edu.



‌



