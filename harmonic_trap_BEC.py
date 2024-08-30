##################################################################
#
# Dicke model
# Huang & Hu, equation (4)
#
##################################################################

import numpy as np
import matplotlib.pyplot as plt

ℏ = 1

class DickeModel:
    def __init__(self, N, ω, Ω, δ, λ, m):
        self.N = N           # Number of qubits
        self.ω = ω   # Frequency of the bosonic mode
        self.Ω = Ω           # Raman coupling strength
        self.δ = δ # detuning from level splitting
        self.λ = λ # wavelength of Raman lasers
        self.m = m # mass
        
        γ = np.sqrt(2) * np.pi * ℏ / (m * λ)
        
        # Create Pauli matrices
        self.sigma_z = np.array([[1, 0], [0, -1]])
        self.sigma_x = np.array([[0, 1], [1, 0]])
        
        # Create identity matrices
        self.I_q = np.eye(2**N)  # Identity for qubits
        self.I_f = np.eye(100)   # Identity for bosonic mode, here using 100 states
        
        # Create bosonic mode operators
        self.a = np.diag(np.sqrt(np.arange(1, 100)), 1)   # Annihilation operator
        self.a_dagger = self.a.T                         # Creation operator
        
        # Tensor product of operators
        self.H0 = ℏ * ω * N * self.a_dagger @ self.a
        #self.Hint = Ω * np.sum([np.kron(np.kron(np.eye(2**k), self.sigma_x), np.kron(np.eye(2**(N-k-1)), (self.a + self.a_dagger))) for k in range(N)], axis=0)
        self.Hint = ℏ * Ω * self.sigma_x + δ * self.sigma_z / 2 + 1j * γ * np.sqrt(2 * m * ℏ * ω) * (self.a_dagger - self.a) @ self.sigma_z
        
        # Total Hamiltonian
        self.H = self.H0 + self.Hint
    
    def ground_state_energy(self):
        """Calculate the ground state energy of the Dicke model."""
        eigenvalues, _ = np.linalg.eigh(self.H)
        return np.min(eigenvalues)
    
    def plot_energy_spectrum(self):
        """Plot the energy spectrum of the Dicke model."""
        eigenvalues, _ = np.linalg.eigh(self.H)
        plt.plot(eigenvalues, marker='o', linestyle='-', color='b')
        plt.xlabel('Eigenvalue index')
        plt.ylabel('Energy')
        plt.title('Energy Spectrum of the Dicke Model')
        plt.grid(True)
        plt.show()

# Example usage:
if __name__ == "__main__":
    N = 2           # Number of qubits
    ω = 1.0     # Frequency of the bosonic mode
    Ω = 0.1         # Coupling strength
    δ = 0
    m = 1
    λ = 1
    
    dicke_model = DickeModel(N, ω, Ω, δ, λ, m)
    
    print(f"Ground state energy: {dicke_model.ground_state_energy()}")
    
    dicke_model.plot_energy_spectrum()