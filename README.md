# General
This repository includes simulations and other components for the investigation of spin squeezing and related topics in Bose-Einstein condensates. The Dicke model is largely functional but is still in development. Examples are provided in the documentation. Everything else is a work in progress. Check comments in the code for documentation.

# Version list
Critical dependencies include Python, matplotlib, and numpy. Convenient dependencies include scipy, tqdm, and qutip. The code must be altered throughout to remove tqdm dependence if desired.

| Package    | Version  |
|------------|----------|
| Python     | 3.11.0   |
| matplotlib | 3.9.2    |
| numpy      | 1.23.5   |
| scipy      | 1.9.3    |
| tqdm       | 4.66.1   |
| qutip      | 5.0.3    |

# Installation
Clone the repository using git or simply copy the desired code directly. The only atypical package I use is tqdm, so install it or adjust the code to remove its dependence.

# Dicke model
This code is based on the typical Dicke model.

$$
H = \omega a^\dagger a + \omega_0 J_z + \frac{g}{\sqrt{N}} \left( a^\dagger + a \right) \left( J_+ + J_- \right)
$$

This might not be the actual Hamiltonian used in the code, so check documentation. The key idea is to illustrate the quantum phase transition for the "normal" phase to the "superradiant" phase, which typically depends on the frequency of the field mode and the frequency of the atomic resonance.
