## Application of Boltzmann Generators for studying of structural phase transitions

This is a fork of code published as supplementary material for research 
article _Boltzmann generators: Sampling equilibrium states of many-body
systems with deep learning_ which is available at 
https://science.sciencemag.org/content/365/6457/eaaw1147.

The original code is accessible at: http://dx.doi.org/10.5281/zenodo.3242635

### Differences between original code and this repository

- Adapted to TensorFlow 2 (original code works only with version 1).
- Refactored and more user-friendly code of Boltzmann generator 
  (eliminated duplicities, one training method that can handle 
  all scenarios instead of multiple trainers, etc).
- Some changes in code that increase its compatibility with PEP coding
  standards.
- Added a documentation for methods and comments in the code that 
  simplify understanding of the code.
- Only code that is truly needed by this project was kept.

### Requirements

- `python` 3.6 or newer
- `tensorflow` 2
- `numpy`
- `scipy`
- `matplotlib`
- `pyemma`