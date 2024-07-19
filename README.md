# NA-MEMD for Python


## Introduction
Python implementation of Noise-Assisted Multivariate Empirical Mode Decomposition (NA-MEMD). Includes functions for noise generation, multivariate empirical mode breakdown (MEMD).  
  
Offers additional options for noise generation, and allows for n-dimensional NA-MEMD as opposed to the previous bivariate option [1].


## Dependencies 
- NumPy
- SciPy  
- sys
- math

```bash
pip install numpy, sys, SciPy, math
```


## Functions and Usage  
to be updated
```python
signal = np.random.randn(5, 1000)
memd_imfs = na_memd(signal)  # traditional memd signal processing
na_memd_imfs = na_memd(signal, method="na_fix")  # na_memd processing with noise assistance
add_noise = add_noise(signal, method = "na_fix")  # add noise only to signal without EMD processing
```


## Acknowledgements
Several existing packages and repositories were referenced in the creation of this library. All credit goes to these authors for their contributions to the field.
* https://github.com/AaronLi43/ginkgo_glasgow [1]
* https://github.com/laszukdawid/PyEMD/tree/master
* https://github.com/mariogrune/MEMD-Python-/tree/master [2]
* https://github.com/STherese/NA-MEMD-for-EEG


## Citations
[1] Y. Zhang, G. Wang, Z. Li, et al., "Matlab Open Source Code: Noise-Assisted Multivariate Empirical Mode Decomposition Based Causal Decomposition for Causality Inference of Bivariate Time Series," Front. Neuroinform., vol. 16, Art. no. 851645, Jun. 2022. doi: 10.3389/fninf.2022.851645  [LINK](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9243260/)  
[2] “Research | Empirical Mode Decomposition (EMD), Multivariate EMD, Matlab code and data sources ∴ Dr. Danilo P. Mandic,” www.commsp.ee.ic.ac.uk. https://www.commsp.ee.ic.ac.uk/~mandic/research/emd.htm (accessed Jul. 09, 2024)  [LINK](https://www.commsp.ee.ic.ac.uk/~mandic/research/emd.htm)
‌



