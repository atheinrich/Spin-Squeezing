########################################################################################################################################################
# Dicke model
########################################################################################################################################################

########################################################################################################################################################
# Summary
""" Overview
    ========
    
    Dicke model
    -----------
    My current understanding is that a†a counts the number of photons absorbed by the atoms from the field,
    and J_z counts the spin energy of the atoms in the z-direction. Without the interaction term,
    the atoms are in the typical ground state, such that they have no excitations. For nonzero coupling
    strengths, the atoms are able to absorb and emit photons with equal probability (?).
    The total spin should be conserved, but that is not apparent in the numerical models to me.

    Examples
    --------
    A) Run examples().
    B) Run examples(N) for some integer N for other automated examples.
    C) Run a customized calculation using the following steps in the command line.
       1)  Set the maximum excitation level (n_max) and the number of atoms (N)
       2)  Generate the Dicke Hamiltonian with init_Dicke_model(N, n_max) using those values
       3a) Plot ground state expectation values as a function of λ using find_occupation()
       3b) Plot all eigenvalues as functions of λ using find_spectrum()

    Works in progress (WIP)
    -----------------------
    other        : try x=c(a†+a) with c → λ_default
    other        : convert everything to OOP """

""" Organization
    ============

    Imports
    -------
    Packages used to streamline calculations and data visualization.
    
    Parameters
    ----------
    Global values not set elsewhere. Mostly defaults, except for J and m_J.
    
    Operators
    ---------
    Construction of global operators.
    
    Operations
    ----------
    Actions made on operators and state vectors.
    
    Algorithms
    ----------
    Series of other functions; intended to streamline typical processes.
    
    Utility
    -------
    Processes that are not specific to the model being simulated.
    
    WIP
    ---
    Functions and algorithms that should not be relied on but have potential value.
    
    Main
    ----
    Ideally only used for algorithm development. """

""" Data descriptions
    =================

    Individual states
    -----------------
    Use the following as an example for the Dicke model, where n_max=3, j=1/2, N=ω=ω0=1, and λ=0:
        |n,  m_J⟩     state array
        |0,  1/2⟩     [1. 0. 0. 0. 0. 0.]            
        |0, -1/2⟩     [0. 1. 0. 0. 0. 0.]            
        |1,  1/2⟩     [0. 0. 1. 0. 0. 0.]            
        |1, -1/2⟩     [0. 0. 0. 1. 0. 0.]            
        |2,  1/2⟩     [0. 0. 0. 0. 1. 0.]            
        |2, -1/2⟩     [0. 0. 0. 0. 0. 1.]            
    Similarly, for N=2:
        |0,    1⟩     [1. 0. 0. 0. 0. 0. 0. 0. 0.]   
        |0,    0⟩     [0. 1. 0. 0. 0. 0. 0. 0. 0.]   
        |0,   -1⟩     [0. 0. 1. 0. 0. 0. 0. 0. 0.]   
        |1,    1⟩     [0. 0. 0. 1. 0. 0. 0. 0. 0.]   
        |1,    0⟩     [0. 0. 0. 0. 1. 0. 0. 0. 0.]   
        |1,   -1⟩     [0. 0. 0. 0. 0. 1. 0. 0. 0.]   
        |2,    1⟩     [0. 0. 0. 0. 0. 0. 1. 0. 0.]   
        |2,    0⟩     [0. 0. 0. 0. 0. 0. 0. 1. 0.]   
        |2,   -1⟩     [0. 0. 0. 0. 0. 0. 0. 0. 1.]   
    Hence, the first N+1 entries correspond to the vacuum state for each possible m_J value.
    The second N+1 entries correspond to one excited state for each possible m_J value.
    The function sort_by_quantum_numbers() can be used to determine initial values for |n, m_J⟩.
    
    Sets of states
    --------------
    Plotting things as a function of λ requires sets of states. These currently have the following structure.
        states       : list(2D_eigenvalue_array, 2D_eigenvector_array)
                       each "row" in these arrays corresponds to a particular λ value
                       each column corresponds to an individual state, where there are n_max*m_J_max states altogether
                       each entry in the "2D" eigenvector array corresponds to an eigenvector   
    For example, this is some states[0] with n_max=2, j=1/2, and N=1 for two λ values.
        [[-5.0 -4.9  5.0  5.1]
         [-5.5 -5.6  5.7  5.6]]
    This is the corresponding states[1] for the same set of parameters.
        [[[ 0.   0.   1.   0. ]
          [ 1.   0.   0.   0. ]
          [ 0.   0.   0.   1. ]
          [ 0.   1.   0.   0. ]]
         [[ 0.2  0.   0.   0.9]
          [ 0.  -0.9  0.2  0. ]
          [ 0.   0.2  0.9  0. ]
          [-0.9  0.   0.   0.2]]]
    The 1D array states[0][0] shows the eigenvalues for the first λ.
    The 2D array states[1][0] shows the eigenvectors for the first λ.
    The eigenvalue entry at states[0][0][2] is 5.0 and corresponds to the eigenvector column at states[1][0][:,2].
    The eigenvalue entry at states[0][1][2] is 5.7 and corresponds to the eigenvector column at states[1][1][:,2]. """

########################################################################################################################################################
# Imports
## Utility
import matplotlib.pyplot as plt                  # plotting
from matplotlib.gridspec import GridSpec         # plotting
from mpl_toolkits.mplot3d import Axes3D          # plotting
from tqdm import tqdm                            # loading bars

## Computation
import numpy as np                               # tensor algebra
from scipy.linalg import expm                    # unitary transformations

########################################################################################################################################################
# Parameters
## Set these here
ω             = 0.01 # field frequency;  single-mode field, like first-order waves in a box (10**16 1/s)
ω0            = 10   # atomic frequency; single frequency for the transition |n⟩ → |n ± 1⟩ for a harmonic oscillator (10**6 1/s)
ℏ             = 1    # Planck's constant (1.054571817 * 10**(-34) J∙s)
m             = 1    # mass

## Set these in main()
n_max_default = 48   # number of energy levels, including vacuum state; sets Fock space
N_default     = 4    # number of particles; sets spin space: integer number of m_J for |m_J ,m_J, ..., m_J⟩

## These are automatic
J             = lambda N: N/2             # total spin of the system: use an integer for bosons
m_J           = lambda N: int(2*J(N) + 1) # dimension of the spin space: counts total number of m_j values
λ_critical    = (ω * ω0)**(1/2)/2         # critical coupling strength

########################################################################################################################################################
# Operators
def create_J_operators(N, individual=False, j_set=None):
    """ Generates total angular momentum matrices in the Fock basis.
        
        Parameters
        ----------
        N          : integer; total number of particles
        individual : string in {'x', 'y', 'z', '+', '-'}; creates a single operator
        
        Returns
        -------
        J_p        : matrix; raising operator for collective angular momentum
        J_m        : matrix; lowering operator for collective angular momentum
        J_x        : matrix; x-component operator for collective angular momentum
        J_y        : matrix; y-component operator for collective angular momentum
        J_z        : matrix; z-component operator for collective angular momentum """
    
    global J, J_p, J_m, J_x, J_y, J_z, J_x_spin, J_z_spin
    
    # Allows for manual setting of j and m_j
    if j_set:
        j         = N * j_set
        dimension = int(round(2 * j_set + 1))
    else:
        j         = J(N)
        dimension = m_J(N)

    if N == 0:
        J = np.eye(dimension)
        return {'x': J, 'y': J, 'z': J, '+': J, '-': J}

    # Set all operators
    if individual == False:
        
        # Ladder operators
        J_p = np.zeros((dimension, dimension))
        J_m = np.zeros((dimension, dimension))
        for i in tqdm(range(dimension - 1), desc=f"{'creating J operators':<35}"):
            m           = j - i
            J_p[i, i+1] = ℏ * np.sqrt(j*(j+1)-m*(m-1))
            J_m[i+1, i] = ℏ * np.sqrt(j*(j+1)-m*(m-1))

        # Component operators
        J_x = (1/2) *(J_p + J_m)
        J_y = (1/2j)*(J_p - J_m)
        J_z = ℏ * np.diag([j-m for m in range(dimension)])
        
        J_x_spin = J_x
        J_z_spin = J_z
        
        return {'x': J_x, 'y': J_y, 'z': J_z, '+': J_p, '-': J_m}
        
    # Raising operator alone
    elif individual == '+':
        J_p = np.zeros((dimension, dimension))
        for i in range(dimension - 1):
            m           = j - i
            J_p[i, i+1] = ℏ * np.sqrt(j*(j+1)-m*(m-1))
        return J_p

    # Lowering operator alone
    elif individual == '-':
        J_m = np.zeros((dimension, dimension))
        for i in range(dimension - 1):
            m           = j - i
            J_m[i+1, i] = ℏ * np.sqrt(j*(j+1)-m*(m-1))
        return J_m

    # J_x operator alone
    elif individual == 'x':
        J_p = np.zeros((dimension, dimension))
        J_m = np.zeros((dimension, dimension))
        for i in range(dimension - 1):
            m           = j - i
            J_p[i, i+1] = ℏ * np.sqrt(j*(j+1)-m*(m-1))
            J_m[i+1, i] = ℏ * np.sqrt(j*(j+1)-m*(m-1))
        return (1/2)*(Jp + Jm)

    # J_y operator alone
    elif individual == 'y':
        J_p = np.zeros((dimension, dimension))
        J_m = np.zeros((dimension, dimension))
        for i in range(dimension - 1):
            m           = j - i
            J_p[i, i+1] = ℏ * np.sqrt(j*(j+1)-m*(m-1))
            J_m[i+1, i] = ℏ * np.sqrt(j*(j+1)-m*(m-1))
        return (1/2j)*(Jp - Jm)
    
    # Create J_z alone
    elif individual == 'z':
        return ℏ * np.diag(np.arange(j, -(j+1), -1))

def create_a_operators(n_max):
    """ Generates creation and annihilation matrices in the Fock basis.
        
        Parameters
        ----------
        n_max : integer; number of excitations allowed per atom
        
        Returns
        -------
        a     : matrix; creation operator for photon field
        a_dag : matrix; annihilation operator for photon field """
    
    global a, a_dag, a_field, a_dag_field
    
    a = np.zeros((n_max, n_max))
    for n in tqdm(range(1, n_max), desc=f"{'creating a operators':<35}"):
        a[n-1, n] = np.sqrt(n)
    a_dag = a.conj().T
    a_field = a
    a_dag_field = a_dag
    
def create_parity_operator():
    """ Generate parity operator that commutes with the standard Dicke model. """
    
    global P
    
    field_parity = expm(1j * np.pi * a_dag_field @ a_field)
    spin_parity  = expm(1j * np.pi * J_z_spin)
    
    P = np.kron(field_parity, spin_parity)
    for i in range(len(P)):
        for j in range(len(P[i])):
            if abs(np.real(P[i][j])) <= 1e-10:
                if abs(np.imag(P[i][j])) <= 1e-10:
                    P[i][j] = 0
                else:
                    P[i][j] = np.imag(P[i][j])
    P = np.real(P)

    return P

def compute_tensor_products(n_max, N):
    """ Takes the tensor product of the field and atom operators and yields the full Hamiltonian.
        
        Parameters
        ----------
        n_max : integer; number of excitations allowed per atom
        N     : integer; total number of particles """
    
    global J_p, J_m, J_x, J_y, J_z, a, a_dag, H_field, H_atom, H_int, H
    
    a     = np.kron(a,     np.eye(m_J(N)))
    a_dag = np.kron(a_dag, np.eye(m_J(N)))
    
    J_p   = np.kron(np.eye(n_max), J_p) # raises total m_j value, but does not alter the number of photons
    J_m   = np.kron(np.eye(n_max), J_m) # lowers total m_j value, but does not alter the number of photons
    J_x   = np.kron(np.eye(n_max), J_x) 
    J_y   = np.kron(np.eye(n_max), J_y) 
    J_z   = np.kron(np.eye(n_max), J_z) # yields the total m_j value, but does not alter the number of photons

    return {'J_x': J_x, 'J_y': J_y, 'J_z': J_z, 'J_m': J_m, 'J_p': J_p, 'a': a, 'a_dag': a_dag}

def Dicke_Hamiltonian(N):
    """ Constructs the Hamiltonian given global operator values.
        
        Parameters
        ----------
        N : integer; total number of particles
        
        Features
        --------
        λ : integer; coupling strength """
    
    global H, H_field, H_atom, H_int

    H_field = ℏ * ω  * (a_dag @ a)         # counts the energy of each photon; (1/2)*np.eye(a.shape[0])
    H_atom  = ℏ * ω0 * J_z                 # counts the energy of each spin
    H_int   = 2 * ℏ / np.sqrt(N) * (a + a_dag) @ J_x   # quantifies the interaction between the atoms and the field
    H       = lambda λ: H_field + H_atom + λ*H_int # sums the total energy and sets the interaction strength

    return H

########################################################################################################################################################
# Operations
def eigenstates(matrix, ground_state=False):
    """ Calculates eigenvalues and eigenvectors for the given matrix.
        For some reason, eigh provides the same eigenvectors as QuTiP, but eig does not.
        
        Parameters
        ----------
        matrix                      : matrix; self-explanatory
        ground_state (deprecated)   : Boolean; returns only the lowest eigenvalue and its eigenvector

        Returns
        -------
        [eigenvalues, eigenvectors] : list of arrays; eigenvalues and eigenvectors
                                      ex. [array(list_of_eigenvalues), array(eigenvectors)] """
    
    eigenvalues, eigenvectors = np.linalg.eig(matrix)     # 1D array, 2D array
    eigenvectors = eigenvectors                         # each row corresponds to a single eigenvector

    # Return ground state eigenvalue and eigenvector
    if ground_state:
        E_min = np.min(eigenvalues)
        ψ0_min = eigenvectors[np.argmin(eigenvalues), :]
        return [np.array([E_min]), np.array([ψ0_min])]                # [1D array, 1D array]
    
    # Return all eigenvalues and eigenvectors
    else: return [eigenvalues, eigenvectors]              # [1D array, 2D array]

def expectation(operator, state, single_state=True):
    """ Just shorthand for some numpy methods. 
        
        Parameters
        ----------
        operator     : 2D array
        state:       : standard or column vector
        single_state : flags as column vector
        
        Returns
        -------
        expectation_value : float; single_state=True yields one number 
        expectation_array : 2D array; single_state=False has one row per λ """
    
    # Single column vector
    if single_state:
        expectation_value = np.conj(state).T @ operator @ state
        return np.real(expectation_value.item())
    
    # Standard data container
    else:
        expectation_array = []
        
        # Sort through trials
        for i in tqdm(range(len(state[1])), desc=f"{'calculating expectation values':<35}"):
            
            # Sort through states
            temp_list_1 = []
            for j in range(len(state[1][i][0])):
                temp_list_2 = expectation(operator, state[1][i][:,j].reshape((state[1][i][:,j].shape[0], 1)))
                temp_list_1.append(temp_list_2)
            expectation_array.append(np.array(temp_list_1).T)
        
        return np.array(expectation_array)

def uncertainty(states, operator):
    """ Calculates the standard deviation of a given operator and a set of states. """

    # Initialize data containers
    expectations, output = [[], []], []
    
    # Calculate expectation values
    expectations[0] = expectation(operator,          states, single_state=False)
    expectations[1] = expectation(operator@operator, states, single_state=False)
    
    # Use expectation values to calculate uncertainty
    for i in range(len(expectations[0])):
        cache = []
        for j in range(len(list(expectations[0][i]))):
            cache.append(np.sqrt(abs(list(expectations[1])[i][j]-list(expectations[0])[i][j]**2)))
        output.append(cache)
    return np.array(output)

def bosonic_squeezing(states):
    """ Calculates the standard deviation of a given operator and a set of states. """

    # Initialize data containers
    expectations, output = [[], [], []], []
    
    # Calculate expectation values
    expectations[0] = expectation(a_dag@a, states, single_state=False)
    expectations[1] = expectation(a,       states, single_state=False)
    expectations[2] = expectation(a@a,     states, single_state=False)
    
    # Use expectation values to calculate uncertainty
    for i in range(len(expectations[0])):
        cache = []
        for j in range(len(list(expectations[0][i]))):
            factor = 1 + 2*(expectations[0][i][j] - abs(expectations[1][i][j])**2 - abs(expectations[2][i][j] - expectations[1][i][j]**2))
            cache.append(factor)
        output.append(cache)
    return np.array(output)

def partial_trace(ρ, dim_A, dim_B, trace_out):
    """ Computes the partial trace of a matrix.

        Parameters
        ----------
        ρ          : 2D array; density matrix 
        dim_A      : integer; dimension of subsystem A
        dim_B      : integer; dimension of subsystem B
        trace_out  : string in {'A', 'B'}; subsystem to be traced out

        Returns
        -------
        ρ_reduced  : reduced matrix after the partial trace """
    
    ρ = ρ.reshape((dim_A, dim_B, dim_A, dim_B))
    if trace_out == 'B':
        ρ_reduced = np.trace(ρ, axis1=1, axis2=3)
    elif trace_out == 'A':
        ρ_reduced = np.trace(ρ, axis1=0, axis2=2)
    return ρ_reduced

########################################################################################################################################################
# Algorithms
def find_occupation(variable_set, states):
    """ Prepare ⟨n⟩ and ⟨J_z⟩ for plotting. """
    
    # Calculate expectation values
    n_expectations   = expectation(a_dag@a, states, single_state=False)
    J_x_expectations = expectation(J_x,     states, single_state=False)
    J_z_expectations = expectation(J_z,     states, single_state=False)
    
    # Construct and return plot list
    plot_list = [[(f"$λ$", f"$⟨n⟩$"),   (variable_set, n_expectations),   (0, 1), ('plot')],
                 [(f"$λ$", f"$⟨J_x⟩$"), (variable_set, J_x_expectations), (1, 0), ('plot')],
                 [(f"$λ$", f"$⟨J_z⟩$"), (variable_set, J_z_expectations), (1, 2), ('plot')]]
    return plot_list

def find_spectrum(variable_set, states):
    """ Prepare energy eigenvalues for plotting.
    
        Optional
        --------
        Subtract the ground state energy from all eigenvalues
            for i in range(len(states[0])):
                for j in range(len(states[0][0])): 
                    val = states[0][i].copy()[0]
                    states[0][i][-(j+1)] -= val """

    return [[(f"$λ$", f"$E$"), (variable_set, states[0]), (0, 0), ('plot')]]

def find_entropy(variable_set, states, dim_A, dim_B):

    def von_Neumann_entropy(ρ, base=2):
        """ Calculate the von Neumann entropy via S(ρ) = -Tr(ρ log ρ).
        
            Parameters
            ----------
            ρ : 2D array; density matrix
            base : int or float; base of the logarithm (default is 2)
            
            Returns
            -------
            entropy : float; von Neumann entropy """
        
        eigvals = np.linalg.eigvalsh(ρ)
        eigvals = eigvals[eigvals > 0]  # avoids issues with log(0) by only considering non-zero eigenvalues
        
        if base == 2:
            log_eigvals = np.log2(eigvals)
        elif base == np.e:
            log_eigvals = np.log(eigvals)
        else:
            log_eigvals = np.log(eigvals) / np.log(base)  # For custom base logarithms
        
        return -np.sum(eigvals * log_eigvals)

    # Prepare entropy arrays
    entropy_tot   = np.zeros_like(states[0])
    entropy_field = np.zeros_like(states[0])
    entropy_spin  = np.zeros_like(states[0])
    eigensum_spin = np.zeros_like(states[0])
    
    # Sort through trials
    for i in tqdm(range(len(states[1])), desc=f"{'calculating entropy':<35}"):
    
        # Sort through states:
        for j in range(len(states[1][0][0])):
            
            # Extract state and compute density matrix
            state = states[1][i][:,j].reshape(states[1][i][:,j].shape[0], 1)
            ρ     = np.outer(state, state.conj())
            
            # Compute reduced density matrices
            #ρ_field = partial_trace(ρ, dim_A, dim_B, trace_out='B')
            ρ_spin  = partial_trace(ρ, dim_A, dim_B, trace_out='A')
            
            eigensum = partial_transpose(ρ)
            
            # Calculate the von Neumann entropy for the total system and subsystems
            entropy_tot[i][j]   = von_Neumann_entropy(ρ,       base=2)
            entropy_field[i][j] = von_Neumann_entropy(ρ_spin,  base=2)
            #entropy_spin[i][j]  = von_Neumann_entropy(ρ_field, base=2)
            eigensum_spin[i][j] = (eigensum - 1)/2
    
    plot_list = [[(f"$λ$", f"$𝒮_Σ$"), (variable_set, entropy_tot),   (0, 0), ('plot')],
                 [(f"$λ$", f"$𝒮_γ$"), (variable_set, entropy_field), (0, 1), ('plot')],
                 [(f"$λ$", f"$(|ρ|^T-1)/2$"), (variable_set, eigensum_spin), (0, 2), ('plot')]]
    return plot_list

def Chebyshift(variable_set, states):
    from scipy.special import jv  # Bessel function of the first kind
    from scipy.sparse import identity, csr_matrix
    from scipy.sparse.linalg import eigsh, LinearOperator
    
    def chebyshev_time_evolution(H, input_state, t, num_terms):
    
        # Estimate the max and min eigenvalues of H
        E_min, E_max = eigsh(H, k=2, which='BE', return_eigenvectors=False)
        
        psi_cache = np.zeros_like(input_state, dtype=np.complex128)
        input_state = psi_cache+input_state

        # Scale the Hamiltonian
        H_scaled = (2 * H - (E_max + E_min) * csr_matrix(identity(H.shape[0]))) / (E_max - E_min)

        # Initial Chebyshev polynomials
        T0 = input_state
        T1 = H_scaled @ input_state
        
        # Time evolution result (initialized with the first term)
        output_state = jv(0, t * (E_max - E_min) / 2) * T0
        
        # Iteratively compute higher-order terms
        for n in range(1, num_terms):
        
            Tn = np.zeros(T0.shape[0], dtype=np.complex128).reshape((input_state.shape[0], 1))
            Tn += 2 * (H_scaled @ T1) - T0

            # Add the contribution of the nth term
            output_state += (2 * (-1j)**n * jv(n, t * (E_max - E_min) / 2)) * Tn
            
            # Update for the next iteration
            T0, T1 = T1, Tn
        
        return output_state
  
    # Set time parameters
    t_max   = 1   # set time interval
    t_shift = 0   # set start time
    dt      = 0.01 # set time steps
    times   = np.linspace(t_shift, t_max+t_shift, int((t_max-t_shift) / dt))

    # Set iteration length
    num_terms = 10

    # Initialize data container
    plot_list = []

    # Cycle through trials
    for i in tqdm(range(len(states[1])), desc=f"{'calculating evolution':<35}"):
        expectation_values = []
        
        # Cycle through states
        for j in range(states[1][i].shape[1]):
            expectation_values.append([])

            # Extract column vector
            state = states[1][i][:,j].reshape((states[1][i][:,j].shape[0], 1))
            
            # Evolve state
            for t in times:
            
                # Compute the time-evolved state at time t
                state_evolved = chebyshev_time_evolution(H(variable_set[i]), state, t, num_terms)
                state_evolved = state_evolved / np.linalg.norm(state_evolved)
                
                # Calculate a property of the evolved state (e.g., probability |state_evolved|^2)
                measure = expectation(H(variable_set[i]), state_evolved, single_state=True)
                
                # Store the total measure at this time step (or any other property of interest)
                expectation_values[j].append(measure)

        expectation_values = np.array(expectation_values).T
        plot_list.append([(f"$t,\tλ={round(variable_set[i],2)}$", f"$⟨E⟩$"), 
                          (times, expectation_values), 
                          (0, i), 
                          ('plot')])
    return plot_list
    
def Lindbladian(variable_set, states):
    global H, J_m

    # Set time parameters
    t_max     = 1   # set time interval
    t_shift   = 0    # set start time
    dt        = 0.01  # set time steps
    times     = np.linspace(t_shift, t_max+t_shift, int((t_max-t_shift) / dt))

    # Set Lindbladian operators
    L = [J_m] # set as [np.eye(J_z.shape[0])] to retain Schrodinger equation

    # Initialize data container for plotting
    plot_list = []

    # Sort through each λ
    for i in tqdm(range(len(variable_set)), desc=f"{'calculating Lindbladian':<35}"):

        # Initialize data container for plotting
        expectation_values = []

        # Generate density matrices
        ρ_array = []
        for j in range(states[1][i].shape[1]):
            ρ_array.append(np.outer(states[1][i][:,j], states[1][i][:,j].conj()))
        ρ_array = np.array(ρ_array)

        # Sort through density matrices
        for j in range(len(ρ_array)):
            ρ = ρ_array[j]
            expectation_values.append([])

            # Sort through each time step
            for t in range(len(times)):

                # Store observable for plotting
                expectation_values[j].append(np.real(np.trace(J_z @ ρ)))

                # Construct the Lindbladian and evolve the density matrix
                dρ = -1j * (H(variable_set[i]) @ ρ - ρ @ H(variable_set[i]))
                for M in L:
                    anticommutator = (M.conj().T @ M) @ ρ + ρ @ (M.conj().T @ M)
                    dρ += M @ (ρ @ M.conj().T) - (1/2) * anticommutator
                ρ = ρ + dt * dρ
                
        expectation_values = np.array(expectation_values).T
        plot_list.append([(f"$t, λ={round(variable_set[i],2)}$", f"$⟨J_z⟩$"), 
                          (times, expectation_values), 
                          (0, i), 
                          ('plot')])
    return plot_list

def init_Dicke_model(n_max, N):
    """ Creates operators and the Hamiltonian for the basic Dicke model with variable coupling strength.
        
        Parameters
        ----------
        n_max      : integer; number of excitations allowed per atom
        N          : integer; total number of particles """
    
    create_J_operators(N)              # creates J_p, J_m, J_x, J_y, and J_z given number of particles 
    create_a_operators(n_max)          # creates a and a_dag given number of available energy levels
    compute_tensor_products(n_max, N)  # updates J_p, J_m, J_x, J_y, J_z, a, and a_dag to the full Hilbert space
    create_parity_operator()           # creates parity operator for sorting
    Dicke_Hamiltonian(N)               # uses global operators to construct the full Hamiltonian

def SEOP_Dicke_model():
    """ Spin-exchange optical pumping model
    
        Hamiltonian
        -----------
        ordered by strength : aI∙S + gμS∙B + μI∙B + μK∙B + γN∙S + aK∙S + bK∙(3R^2-1)∙S
        ordered by glamour  : (aI + γN + aK)∙S + (gμS + μI + μK)∙B + bK∙(3R^2-1)∙S """
    
    global a, a_dag, H, P, S_z
    
    # Initialize parameters
    n_max = 48
    N_I   = 4
    N_S   = 4
    N     = N_S + N_I
    
    I     = 3/2
    S     = 1/2

    # Create spin operators
    I_dict = create_J_operators(N_I, individual=False, j_set=I)
    S_dict = create_J_operators(N_S, individual=False, j_set=S)

    I_p   = np.kron(np.eye(n_max), np.kron(np.eye(S_dict['z'].shape[0]), I_dict['+']))
    I_m   = np.kron(np.eye(n_max), np.kron(np.eye(S_dict['z'].shape[0]), I_dict['-']))
    I_x   = np.kron(np.eye(n_max), np.kron(np.eye(S_dict['z'].shape[0]), I_dict['x'])) 
    I_y   = np.kron(np.eye(n_max), np.kron(np.eye(S_dict['z'].shape[0]), I_dict['y'])) 
    I_z   = np.kron(np.eye(n_max), np.kron(np.eye(S_dict['z'].shape[0]), I_dict['z']))
    
    S_p   = np.kron(np.eye(n_max), np.kron(S_dict['+'], np.eye(I_dict['z'].shape[0])))
    S_m   = np.kron(np.eye(n_max), np.kron(S_dict['-'], np.eye(I_dict['z'].shape[0])))
    S_x   = np.kron(np.eye(n_max), np.kron(S_dict['x'], np.eye(I_dict['z'].shape[0]))) 
    S_y   = np.kron(np.eye(n_max), np.kron(S_dict['y'], np.eye(I_dict['z'].shape[0]))) 
    S_z   = np.kron(np.eye(n_max), np.kron(S_dict['z'], np.eye(I_dict['z'].shape[0])))

    # Create field operators
    create_a_operators(n_max)
    a     = np.kron(a,     np.kron(np.eye(S_dict['z'].shape[0]), np.eye(I_dict['z'].shape[0])))
    a_dag = np.kron(a_dag, np.kron(np.eye(S_dict['z'].shape[0]), np.eye(I_dict['z'].shape[0])))

    # Create parity operator
    a_exp        = expm(1j * np.pi * a_dag_field @ a_field)
    I_exp        = expm(1j * np.pi * I_dict['z'])
    S_exp        = expm(1j * np.pi * S_dict['z'])    
    P = np.kron(a_exp, np.kron(S_exp, np.eye(I_exp.shape[0])))
    for i in range(len(P)):
        for j in range(len(P[i])):
            if abs(np.real(P[i][j])) <= 1e-10:
                if abs(np.imag(P[i][j])) <= 1e-10:
                    P[i][j] = 0
                else:
                    P[i][j] = np.imag(P[i][j])
    P = np.real(P)

    # Create Hamiltonian
    H_field = ℏ * ω  * (a_dag @ a)
    H_I     = ℏ * ω0 * I_z
    H_S     = ℏ * ω0 * S_z
    H_spin  = ℏ * ω0 * I_z @ S_z
    H_int   = 2 * ℏ / np.sqrt(N) * (a + a_dag) @ S_x
    H       = lambda λ: H_field + H_I + H_S + H_spin + λ*H_int

    # Generate all eigenstates and eigenvalues
    variable_set = np.linspace(0, 10*λ_critical, 101)
    states       = calculate_states(variable_set)

    # Sort eigenstates and eigenvalues
    sort_dict = {'P': P, 'E': H, 'S_z': S_z}
    states, quantum_numbers = sort_by_quantum_numbers(states, sort='P', secondary_sort='E')

    # Define custom plotting
    def SEOP_occupation(variable_set, states, quantum_numbers):
    
        # Select specific eigenstates
        selected_states = [0, int(len(states[0][0])/2)]
        states          = [states[0][:,selected_states], states[1][:,:,selected_states]]
        quantum_numbers = quantum_numbers[:,selected_states]
    
        n_expectations   = expectation(a_dag@a, states, single_state=False)
        J_x_expectations = expectation(I_z,     states, single_state=False)
        J_z_expectations = expectation(S_z,     states, single_state=False)
        
        plot_list = [[(f"$λ$", f"$⟨n⟩$"),   (variable_set, n_expectations),   (0, 1), ('plot')],
                     [(f"$λ$", f"$⟨I_z⟩$"), (variable_set, J_x_expectations), (1, 0), ('plot')],
                     [(f"$λ$", f"$⟨J_z⟩$"), (variable_set, J_z_expectations), (1, 2), ('plot')]]
        return plot_list
    
    # Make a calculation
    spectrum_plot_list   = find_spectrum(variable_set, states, quantum_numbers)
    occupation_plot_list = SEOP_occupation(variable_set, states, quantum_numbers)
    plot_results(spectrum_plot_list,   quantum_numbers)
    plot_results(occupation_plot_list, quantum_numbers)

def two_spin_Dicke_model():
    """ Spin-exchange optical pumping model
    
        Hamiltonian
        -----------
        ordered by strength : aI∙S + gμS∙B + μI∙B + μK∙B + γN∙S + aK∙S + bK∙(3R^2-1)∙S
        ordered by glamour  : (aI + γN + aK)∙S + (gμS + μI + μK)∙B + bK∙(3R^2-1)∙S """
    
    global a, a_dag, H, P, S_z
    
    # Initialize parameters
    n_max = 64
    N_I   = 1
    N_S   = 7
    N     = N_S + N_I
    
    I     = 1/2
    S     = 1/2

    # Create spin operators
    I_dict = create_J_operators(N_I, individual=False, j_set=I)
    S_dict = create_J_operators(N_S, individual=False, j_set=S)

    I_p   = np.kron(np.eye(n_max), np.kron(np.eye(S_dict['z'].shape[0]), I_dict['+']))
    I_m   = np.kron(np.eye(n_max), np.kron(np.eye(S_dict['z'].shape[0]), I_dict['-']))
    I_x   = np.kron(np.eye(n_max), np.kron(np.eye(S_dict['z'].shape[0]), I_dict['x'])) 
    I_y   = np.kron(np.eye(n_max), np.kron(np.eye(S_dict['z'].shape[0]), I_dict['y'])) 
    I_z   = np.kron(np.eye(n_max), np.kron(np.eye(S_dict['z'].shape[0]), I_dict['z']))
    
    S_p   = np.kron(np.eye(n_max), np.kron(S_dict['+'], np.eye(I_dict['z'].shape[0])))
    S_m   = np.kron(np.eye(n_max), np.kron(S_dict['-'], np.eye(I_dict['z'].shape[0])))
    S_x   = np.kron(np.eye(n_max), np.kron(S_dict['x'], np.eye(I_dict['z'].shape[0]))) 
    S_y   = np.kron(np.eye(n_max), np.kron(S_dict['y'], np.eye(I_dict['z'].shape[0]))) 
    S_z   = np.kron(np.eye(n_max), np.kron(S_dict['z'], np.eye(I_dict['z'].shape[0])))

    # Create field operators
    create_a_operators(n_max)
    a     = np.kron(a,     np.kron(np.eye(S_dict['z'].shape[0]), np.eye(I_dict['z'].shape[0])))
    a_dag = np.kron(a_dag, np.kron(np.eye(S_dict['z'].shape[0]), np.eye(I_dict['z'].shape[0])))

    # Create parity operator
    a_exp        = expm(1j * np.pi * a_dag_field @ a_field)
    I_exp        = expm(1j * np.pi * I_dict['z'])
    S_exp        = expm(1j * np.pi * S_dict['z'])    
    P = np.kron(a_exp, np.kron(S_exp, I_exp))
    for i in range(len(P)):
        for j in range(len(P[i])):
            if abs(np.real(P[i][j])) <= 1e-10:
                if abs(np.imag(P[i][j])) <= 1e-10:
                    P[i][j] = 0
                else:
                    P[i][j] = np.imag(P[i][j])
    P = np.real(P)

    # Create Hamiltonian
    H_field = ℏ * ω  * (a_dag @ a)
    H_spin  = ℏ * ω0 * (S_z + I_z)
    H_int   = 2 * ℏ / np.sqrt(N) * (a + a_dag) @ (S_x - I_x)
    H       = lambda λ: H_field + H_spin + λ*H_int

    # Generate all eigenstates and eigenvalues
    variable_set = np.linspace(0, 5*λ_critical, 101)
    states       = calculate_states(variable_set)

    # Sort eigenstates and eigenvalues
    sort_dict = {'P': P, 'E': H, 'S_z': S_z, 'n': a_dag@a}
    states, quantum_numbers = sort_by_quantum_numbers(states, sort='P', secondary_sort='E')

    # Define custom plotting
    def plot_n_S(variable_set, states, quantum_numbers):
    
        # Select specific eigenstates
        selected_states = [0, int(len(states[0][0])/2)]
        states          = [states[0][:,selected_states], states[1][:,:,selected_states]]
        quantum_numbers = quantum_numbers[:,selected_states]
    
        n_expectations   = expectation(a_dag@a, states, single_state=False)
        J_x_expectations = expectation(I_z,     states, single_state=False)
        J_z_expectations = expectation(S_z,     states, single_state=False)
        
        plot_results([[(f"$λ$", f"$⟨n⟩$"),   (variable_set, n_expectations),   (0, 1), ('plot')],
                      [(f"$λ$", f"$⟨I_z⟩$"), (variable_set, J_x_expectations), (1, 0), ('plot')],
                      [(f"$λ$", f"$⟨J_z⟩$"), (variable_set, J_z_expectations), (1, 2), ('plot')]],
                      quantum_numbers = quantum_numbers)
    
    # Make a calculation
    #find_spectrum(variable_set, states, quantum_numbers)
    plot_n_S(variable_set, states, quantum_numbers)

def examples(specific_example=0):
    """ Run a preset example.
    
        Parameters
        ----------
        specific_example : natural number
                           0) Introduction; plot a simple spectrum
                           1) Under the hood; plot ⟨n⟩ and ⟨J⟩ for the ground state of each parity
                           2) Phase transition; plot a slightly more complicated spectrum
                           3) Crossing; plot an even more complicated spectrum
                           4) Bifurcations; plot a find_spectrum with a non-Hermitian Hamiltonian """
    
    global ω, ω0, J_z
    
    # Example 0: simple spectrum
    if specific_example == 0:
    
        # Set parameters
        ω, ω0      = 1, 1
        n_max, N   = 2, 1
        λ_critical = (ω * ω0)**(1/2)/2
        print_parameters(ω, ω0, n_max, N)
    
        # Initialize model
        init_Dicke_model(n_max, N)

        # Generate eigenstates and eigenvalues
        variable_set = np.linspace(1e-10, 3*λ_critical, 101)
        states       = calculate_states(variable_set)

        # Sort eigenstates and eigenvalues
        states, quantum_numbers = sort_by_quantum_numbers(states, sort='P', secondary_sort='E')

        # Make a calculation
        plot_list = find_spectrum(variable_set, states)
        plot_results(plot_list, quantum_numbers)
    
    # Example 1: cavity occupation
    elif specific_example == 1:
    
        # Set parameters
        ω, ω0      = 0.1, 10
        n_max, N   = 24,  2
        λ_critical = (ω * ω0)**(1/2)/2
        print_parameters(ω, ω0, n_max, N)

        # Initialize model
        init_Dicke_model(n_max=24, N=2)

        # Generate eigenstates and eigenvalues
        variable_set = np.linspace(1e-10, 3*λ_critical, 101)
        states       = calculate_states(variable_set)

        # Sort eigenstates and eigenvalues
        states, quantum_numbers = sort_by_quantum_numbers(states, sort='P', secondary_sort='E')

        # Select specific eigenstates
        states, quantum_numbers = select_states(variable_set, states, quantum_numbers, selection="ground")

        # Make a calculation
        plot_list = find_occupation(variable_set, states)
        plot_results(plot_list, quantum_numbers)
    
    # Example 2: dense spectrum
    elif specific_example == 2:
    
        # Set parameters
        ω, ω0      = 0.1, 10
        n_max, N   = 48,  4
        λ_critical = (ω * ω0)**(1/2)/2
        print_parameters(ω, ω0, n_max, N)
        
        # Initialize model
        init_Dicke_model(n_max=48, N=4)

        # Generate eigenstates and eigenvalues
        variable_set = np.linspace(1e-10, 3*λ_critical, 101)
        states       = calculate_states(variable_set)

        # Sort eigenstates and eigenvalues
        states, quantum_numbers = sort_by_quantum_numbers(states, sort='P', secondary_sort='E')

        # Make a calculation
        plot_list = find_spectrum(variable_set, states)
        plot_results(plot_list, quantum_numbers)
    
    # Example 3: resonance (avoided crossings)
    elif specific_example == 3:
    
        # Set parameters
        ω, ω0      = 1,  1
        n_max, N   = 48, 4
        λ_critical = (ω * ω0)**(1/2)/2
        print_parameters(ω, ω0, n_max, N)
        
        # Initialize model
        init_Dicke_model(n_max=48, N=4)

        # Generate eigenstates and eigenvalues
        variable_set = np.linspace(1e-10, λ_critical, 101)
        states       = calculate_states(variable_set)

        # Sort eigenstates and eigenvalues
        states, quantum_numbers = sort_by_quantum_numbers(states, sort='P', secondary_sort='E')

        # Make a calculation
        plot_list = find_spectrum(variable_set, states)
        plot_results(plot_list, quantum_numbers)
    
    # Example 4: bifurcations
    elif specific_example == 4:
    
        global H

        # Set parameters
        ω, ω0      = 0.1, 10
        n_max, N   = 24,  2
        λ_critical = (ω * ω0)**(1/2)/2
        print_parameters(ω, ω0, n_max, N)
        
        # Initialize model
        create_J_operators(N)
        create_a_operators(n_max)
        compute_tensor_products(n_max, N)
        create_parity_operator()
        H_field = ℏ * ω  * (a_dag @ a + a @ a)
        H_atom  = ℏ * ω0 * J_z
        H_int   = 2 * ℏ / np.sqrt(N) * (a + a_dag) @ J_x
        H       = lambda λ: H_field + H_atom + λ*H_int

        # Generate eigenstates and eigenvalues
        variable_set = np.linspace(1e-10, 10*λ_critical, 101)
        states       = calculate_states(variable_set)

        # Sort eigenstates and eigenvalues
        states, quantum_numbers = sort_by_quantum_numbers(states, sort='P', secondary_sort='E')

        # Make a calculation
        plot_list = find_spectrum(variable_set, states)
        plot_results(plot_list, quantum_numbers)
    
    # Example 5: Chebyshev evolution
    elif specific_example == 5:
    
        # Set parameters
        ω, ω0      = 0.1, 10
        n_max, N   = 12,  2
        λ_critical = (ω * ω0)**(1/2)/2
        print_parameters(ω, ω0, n_max, N)
        
        # Initialize model
        init_Dicke_model(n_max, N)

        # Generate eigenstates and eigenvalues
        variable_set = np.linspace(1e-10, 2*λ_critical, 11)
        states       = calculate_states(variable_set)

        # Sort eigenstates and eigenvalues
        states, quantum_numbers = sort_by_quantum_numbers(states, sort='P', secondary_sort='E')

        # Select specific eigenstates
        states, quantum_numbers = select_states(variable_set, states, quantum_numbers, selection="random")

        # Make a calculation
        plot_list = Chebyshift(variable_set, states)
        plot_results(plot_list, quantum_numbers, plot_mode="3D")

    # Example 6: Lindbladian evolution
    elif specific_example == 6:
    
        # Set parameters
        ω, ω0    = 0.1, 10
        n_max, N = 24,  2
        λ_critical = (ω * ω0)**(1/2)/2
        print_parameters(ω, ω0, n_max, N)
        
        # Initialize model
        init_Dicke_model(n_max, N)

        # Generate eigenstates and eigenvalues
        variable_set = np.linspace(1e-10, 2*λ_critical, 11)
        states       = calculate_states(variable_set)

        # Sort eigenstates and eigenvalues
        states, quantum_numbers = sort_by_quantum_numbers(states, sort='P', secondary_sort='E')

        # Select specific eigenstates
        states, quantum_numbers = select_states(variable_set, states, quantum_numbers, selection="ground")

        # Make a calculation
        plot_list = Lindbladian(variable_set, states)
        plot_results(plot_list, quantum_numbers, plot_mode="3D")

    # Example 7: SEOP
    elif specific_example == 7:
        SEOP_Dicke_model()

    # Example 8: spin squeezing
    elif specific_example == 8:
    
        # Set parameters
        ω, ω0      = 0.1, 10
        n_max, N   = 32, 32
        λ_critical = (ω * ω0)**(1/2)/2
        print_parameters(ω, ω0, n_max, N)
        
        # Initialize model
        init_Dicke_model(n_max, N)

        # Generate all eigenstates and eigenvalues
        variable_set = np.linspace(1e-10, 3*λ_critical, 101)
        states       = calculate_states(variable_set)

        # Sort eigenstates and eigenvalues
        states, quantum_numbers = sort_by_quantum_numbers(states, sort='P', secondary_sort='E')
        
        # Select specific eigenstates
        states, quantum_numbers = select_states(variable_set, states, quantum_numbers, selection="ground")
        
        # Make a calculation
        ΔJ_x      = uncertainty(states, J_x)
        ΔJ_y      = uncertainty(states, J_y)
        ΔJ_z      = uncertainty(states, J_z)
        J_x_exp   = expectation(J_x, states, single_state=False)
        J_x_exp   = tolerance_check(J_x_exp)
        J_y_exp   = expectation(J_y, states, single_state=False)
        J_z_exp   = expectation(J_z, states, single_state=False)
        product_1 = ΔJ_x * ΔJ_y
        product_2 = ΔJ_y * ΔJ_z
        product_3 = ΔJ_x * ΔJ_z
        ζ         = bosonic_squeezing(states)
        
        plot_list = [[(f"", f"$ζ^2$"),      (variable_set, ζ),         (0, 1), ('plot')],
                     [(f"", f"$⟨J_x⟩$"),     (variable_set, J_x_exp),   (1, 0), ('plot')],
                     [(f"", f"$⟨J_y⟩$"),     (variable_set, J_y_exp),   (1, 1), ('plot')],
                     [(f"", f"$⟨J_z⟩$"),     (variable_set, J_z_exp),   (1, 2), ('plot')],
                     [(f"", f"$ΔJ_x$"),     (variable_set, ΔJ_x),      (2, 0), ('plot')],
                     [(f"", f"$ΔJ_y$"),     (variable_set, ΔJ_y),      (2, 1), ('plot')],
                     [(f"", f"$ΔJ_z$"),     (variable_set, ΔJ_z),      (2, 2), ('plot')],
                     [(f"", f"$ΔJ_xΔJ_y$"), (variable_set, product_1), (3, 0), ('plot')],
                     [(f"", f"$ΔJ_yΔJ_z$"), (variable_set, product_2), (3, 1), ('plot')],
                     [(f"", f"$ΔJ_xΔJ_z$"), (variable_set, product_3), (3, 2), ('plot')]]
        plot_results(plot_list, quantum_numbers)

    # Example 9: entropy
    elif specific_example == 9:
    
        # Set parameters
        ω, ω0      = 0.1,  10
        n_max, N   = 32, 32
        λ_critical = (ω * ω0)**(1/2)/2
        print_parameters(ω, ω0, n_max, N)
        
        # Initialize model
        init_Dicke_model(n_max, N)

        # Generate all eigenstates and eigenvalues
        variable_set = np.linspace(0, 6*λ_critical, 1001)
        states       = calculate_states(variable_set)

        # Sort eigenstates and eigenvalues
        states, quantum_numbers = sort_by_quantum_numbers(states, sort='P', secondary_sort='E')

        # Select specific eigenstates
        states, quantum_numbers = select_states(variable_set, states, quantum_numbers, selection=[0, 1, 2])
        
        # Make a calculation
        plot_list = find_entropy(variable_set, states, n_max, m_J(N))
        plot_results(plot_list, quantum_numbers)

    else:
        print('There are no examples with this value.')

########################################################################################################################################################
# Utility
def calculate_states(variable_set, ground_state=False):
    """ Computes states in the standard representation.
        See data descriptions in Summary for more details. """
    
    eigenvalues, eigenvectors = [], []
    for λ in tqdm(variable_set, desc=f"{'finding eigenstates':<35}"):
        eigenvalue, eigenvector = eigenstates(H(λ), ground_state)  # Compute once
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)  
    return [np.array(eigenvalues), np.array(eigenvectors)] 

def select_states(variable_set, states, quantum_numbers, selection="ground"):
    """ Custructs sets of states from a set of eigenstates.
    
        Parameters
        ----------
        variable_set    : 1D array; typically a range of coupling strengths
        states          : 3D array; standard representation
        quantum_numbers : 3D array; standard representation
        selection       : list or string
                          list:   [<index for each state by sorted eigenvalue>]
                          strings "ground" yields the ground state for each parity
                                  "random" yields a single state as a weighted superposition of eigenstates
     
        Returns
        -------
        states          : 3D array; standard representation
        quantum_numbers : 3D array; standard representation """
    
    # Manual state selection
    if type(selection) == list:
    
        # Select eigenstates by index under given sorting (usually energy)
        selected_states = selection
        
        # Update and return states
        states          = [states[0][:,selected_states], states[1][:,:,selected_states]]
        quantum_numbers = quantum_numbers[:,selected_states]
        return states, quantum_numbers
    
    # Ground states
    elif selection == "ground":
    
        # Choose ground state for each parity
        selected_states = [0, int(len(states[0][0])/2)] # [0, 1, int(len(states[0][0])/2), int(len(states[0][0])/2)+1]
        
        # Update and return states
        states          = [states[0][:,selected_states], states[1][:,:,selected_states]]
        quantum_numbers = quantum_numbers[:,selected_states]
        return states, quantum_numbers
    
    # Random state
    elif selection == "random":
    
        # Initialize data container
        new_states = [[], []]
        
        # Initialize randomization
        seed               = 1
        rng                = np.random.default_rng(seed)
        
        # Choose how many eigenstates to include
        num_eigenstates    = rng.integers(low=1, high=states[0][0].shape[0])
        
        # Choose random eigenstates and weights for each eigenstate
        random_eigenstates = rng.integers(low=0, high=states[0][0].shape[0], size=num_eigenstates)
        random_weights     = rng.uniform(0, 1, num_eigenstates)
        
        # Sort through trials
        for i in range(len(states[1])):
        
            # Construct state
            new_state = np.zeros_like(states[1][0][:,0])
            for j in range(num_eigenstates):
                new_state += random_weights[j] * states[1][i][:,random_eigenstates[j]]
            
            # Normalize and recast as column vector
            new_state = (new_state / np.linalg.norm(new_state)).reshape((new_state.shape[0], 1))
            
            # Calculate energy
            new_energy = expectation(H(variable_set[i]), new_state)
            
            # Append to data container
            new_states[0].append([new_energy])
            new_states[1].append(new_state)
        
        new_states = [np.array(new_states[0]), np.array(new_states[1])]
        return new_states, None

def sort_by_quantum_numbers(states, sort=None, secondary_sort=None, sort_dict={}):
    """ Find quantum numbers for each eigenstate at λ=0, assuming H has been constructed.
    
    Parameters
    ----------
    states         : 2D array; standard representation
    sort           : string in ['P', 'n', 'J_z', 'E']; sets first quantum number to sort by
    secondary_sort : string in ['P', 'n', 'J_z', 'E']; sets second quantum number to sort by

    Returns
    -------
    states         : 2D array; standard representation """
    
    # Initialize some lists
    if not sort_dict:
        parameter_dict = {'P': 0, 'n': 1, 'J_z': 2, 'E': 3}
    else:
        parameter_dict = {}
        for i in range(len(sort_dict)):
            parameter_dict[sort_dict[i]] = i
    
    sorted_states_0     = [] # For sorted eigenvalues
    sorted_states_1     = [] # For sorted eigenstates
    sorted_expectations = [] # For sorted expectations

    # Find n and m_J for each state
    expectation_list = calculate_quantum_numbers(states, sort=sort, secondary_sort=secondary_sort, sort_dict=sort_dict)

    # Loop over each set of states
    for i in range(len(expectation_list)):
        row = expectation_list[i]
        
        # Sort by secondary eigenvalue parameter first (if provided)
        if secondary_sort:
            sorted_indices = np.argsort(row[:, 1], kind='stable')
        else:
            sorted_indices = np.arange(len(row))  # Default indices if no secondary sort
        
        # Then apply a stable sort by the primary eigenvalue parameter, preserving secondary order
        sorted_indices = sorted_indices[np.argsort(row[sorted_indices, parameter_dict[sort]], kind='stable')]

        sorted_row = row[sorted_indices]  # Sort expectation_list row
        sorted_states_0.append(np.array(states[0][i])[sorted_indices])  # Sort eigenvalues in states[0][i]
        sorted_states_1.append(states[1][i][:, sorted_indices])  # Sort states based on sorted indices
        sorted_expectations.append(sorted_row)  # Store sorted expectations

    sorted_states_0 = np.array(sorted_states_0)  # Convert list to array for consistency
    sorted_states_1 = np.array(sorted_states_1)  # Convert list to array for consistency

    return [sorted_states_0, sorted_states_1], np.array(sorted_expectations)

def calculate_quantum_numbers(states, precision=10, sort=None, secondary_sort=None, sort_dict={}):
    """ Find quantum numbers for each eigenstate at λ=0, assuming H has been constructed.
    
        Parameters
        ----------
        states           : 2D array; standard representation
        precision        : integer; sets rounding to control sorting precision
        sort             : string in ['P', 'n', 'J_z', 'E']; sets first quantum number to sort by
        secondary_sort   : string in ['P', 'n', 'J_z', 'E']; sets second quantum number to sort by
    
        Returns
        ------
        expectation_list : 2D array; contains n and m_J for each state """
    
    # Import sorting parameters or try default
    if not sort_dict:
        sort_dict = {'P': P, 'n': a_dag@a, 'J_z': J_z}
    
    # Set rounding
    set_precision = precision
    
    # Cycle through each λ
    expectation_list = []
    for i in tqdm(range(len(states[1])), desc=f"{'calculating numbers':<35}"):
        expectations_rounded = []
        
        # Calculate all quantum numbers (|P, n, J_z, E⟩)
        if sort == None:
            P_expectations   = [expectation(P,         states[1][i][:,j].reshape((states[1][i][:,j].shape[0], 1))) for j in range(len(states[1][0][0]))]
            n_expectations   = [expectation(a_dag @ a, states[1][i][:,j].reshape((states[1][i][:,j].shape[0], 1))) for j in range(len(states[1][0][0]))]
            J_z_expectations = [expectation(J_z,       states[1][i][:,j].reshape((states[1][i][:,j].shape[0], 1))) for j in range(len(states[1][0][0]))]
            E_expectations   = states[0][i]

            for k in range(len(states[0][i])):
                expectations_rounded.append([round(P_expectations[k],   set_precision),
                                             round(n_expectations[k],   set_precision),
                                             round(J_z_expectations[k], set_precision),
                                             round(E_expectations[k],   set_precision)])
            expectation_list.append(np.array(expectations_rounded))
        
        # Calculate one or two quantum numbers
        else:

            # Calculate one quantum number (|sort⟩)
            if secondary_sort == None:
                
                # Avoid recalculating energy values
                if sort == 'E':
                    for k in range(len(states[0][i])):
                        expectations_rounded.append([round(states[0][i][k], set_precision)])
                    expectation_list.append(np.array(expectations_rounded))
                
                # Calculate quantum number
                else:
                    expectations_cache = [expectation(sort_dict[sort], states[1][i][:,j].reshape((states[1][i][:,j].shape[0], 1))) for j in range(len(states[1][0][0]))]
                    for k in range(len(states[0][i])):
                        expectations_rounded.append([round(expectations_cache[k], set_precision)])
                    expectation_list.append(np.array(expectations_rounded))
            
            # Calculate two quantum numbers (|sort, secondary_sort⟩)
            else:
                
                # Calculate first number
                if sort == 'E':
                    expectations_cache_1 = states[0][i]
                else:
                    expectations_cache_1 = [expectation(sort_dict[sort], states[1][i][:,j].reshape((states[1][i][:,j].shape[0], 1))) for j in range(len(states[1][0][0]))]
                
                # Calculate second number
                if secondary_sort == 'E':
                    expectations_cache_2 = states[0][i]
                else:
                    expectations_cache_2 = [expectation(sort_dict[secondary_sort], states[1][i][:,j].reshape((states[1][i][:,j].shape[0], 1))) for j in range(len(states[1][0][0]))]
                for k in range(len(states[0][i])):
                    expectations_rounded.append([round(np.real(expectations_cache_1[k]), set_precision),
                                                 round(np.real(expectations_cache_2[k]), set_precision)])
                expectation_list.append(np.array(expectations_rounded))
    
    return np.array(expectation_list)

def plot_results(results, quantum_numbers=None, plot_mode="2D"):
    """ Plots input data using GridSpec.
        
        Parameters
        ----------
        results         : list of lists; [[titles_1, values_1, indices_1, style_1],
                                          [titles_2, values_2, indices_2, style_2], ...]
        quantum_numbers : 3D array; corresponding quantum numbers for coloring plots
      
        Details
        -------
        titles  : (row_title, column_title); ex. ('x', 'f(x)')
        values  : (x_values,  y_values);     ex. (x_array, y_array)
        indices : (row_index, column_index); ex. (0, 2)
        style   : (plot_type);               ex. ('plot') or ('contour') """
    
    # 2D plotting
    if plot_mode == "2D":
    
        # Initialize grid size based on 2D plot indices
        try:
            max_row = max(item[2][0] for item in results) + 1
            max_col = max(item[2][1] for item in results) + 1
        except:
            max_row = 1
            max_col = 1
        
        # Initialize figure with GridSpec
        fig = plt.figure(figsize=(3*max_col, 3*max_row))
        gs  = GridSpec(max_row, max_col, figure=fig)
        
        # Define color map
        color_map = {1: 'Reds_r', -1: 'Blues_r'}
        def get_colormap_color(cmap_name, column, num_columns):
            cmap = plt.get_cmap(cmap_name)
            normalized_value = column / num_columns
            capped_value = min(normalized_value, 0.9)
            return cmap(capped_value)  # Returns RGBA color

        # Loop through results for individual 2D plots
        for labels, values, indices, style in results:

            ax = fig.add_subplot(gs[indices[0], indices[1]])
            ax.set_xlabel(labels[0], fontsize=16)
            ax.set_ylabel(labels[1], fontsize=16)
            ax.ticklabel_format(useOffset=False)

            if quantum_numbers is not None:
                for i in range(len(values[1][0])):
                    color_category = color_map.get(int(quantum_numbers[0][i][0]))
                    set_color = get_colormap_color(color_category, i+1, len(values[1][0]))

                    if style == 'plot':
                        ax.plot(values[0], values[1][:, i], color=set_color)
                    elif style == 'contour':
                        ax.contourf(values[0], values[1][:, i], values[2], 100)
            else:
                if style == 'plot':
                    ax.plot(values[0], values[1])
                elif style == 'contour':
                    ax.contourf(values[0], values[1], values[2], 100)

        plt.tight_layout()
        plt.show()
    
    # 3D plotting
    elif plot_mode == "3D":
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for i, (labels, values, indices, style) in enumerate(results):
            x = values[0]
            y = np.full_like(values[1], i)
            z = values[1]

            for trial in range(z.shape[1]):
                ax.plot(x, y[:, trial], z[:, trial])

        ax.set_xlabel(f"$t$", fontsize=12)
        ax.set_zlabel(labels[1], fontsize=12)
        plt.show()
    
    print()

def print_parameters(ω, ω0, n_max, N):
    print(f"\n------------------------------------------\n"
          f"{'field frequency':<35}: {ω:<4}|\n"
          f"{'atomic frequency':<35}: {ω0:<4}|\n"
          f"{'critical coupling':<35}: {(ω * ω0)**(1/2)/2:<4}|\n"
          f"{'number of modes':<35}: {n_max:<4}|\n"
          f"{'number of particles':<35}: {N:<4}|\n"
          f"------------------------------------------\n")

def tolerance_check(array, tolerance=1e-8):
    """ Sends each value in an array to zero if it is less than a set tolerance. """

    tolerance = 1e-8
    array[array <= tolerance] = 0
    
    return array

########################################################################################################################################################
# WIP
def partial_transpose(ρ):
    """
    Perform partial transposition on the density matrix `ρ`.
    
    Args:
    - ρ (numpy.ndarray): The density matrix of the bipartite system.
    - dims (tuple): A tuple (dimA, dimB) where dimA is the dimension of the first subsystem 
                    and dimB is the dimension of the second subsystem.
    - subsystem (int): The subsystem to apply the transposition on (0 or 1). 
                       0 for the first subsystem, 1 for the second subsystem.
    
    Returns:
    - numpy.ndarray: The density matrix after partial transposition.
    """
    
    n = ρ.shape[0] // 2
    A, B, C, D = ρ[:n, :n].T, ρ[:n, n:].T, ρ[n:, :n].T, ρ[n:, n:].T
    
    top = np.hstack((A, B))
    bottom = np.hstack((C, D))
    ρ = np.vstack((top, bottom))
    
    # Check for entanglement by computing the eigenvalues
    eigenvalues = np.linalg.eigvals(ρ)
    eigensum    = 0
    for i in range(len(eigenvalues)):
        eigensum += abs(eigenvalues[i])

    # If any eigenvalue is negative, the state is entangled
    #if np.any(eigenvalues < 0):
    #    print("The state is entangled.")
    #else:
    #    print("The state is separable.")
    
    return eigensum

class System:
    def __init__(self, ω=1, ω0=1, n_max=32, N=32, qs=True):
        """ Initializes operators and parameters.
            Parameters
            ----------
            qs : Boolean; quick start sets a default variable and finds states """
        
        # Initialize parameters
        self.ω     = ω
        self.ω0    = ω0
        self.n_max = n_max
        self.N     = N
        self.crit  = (ω * ω0)**(1/2)/2
        
        # Create operators
        create_J_operators(N)
        create_a_operators(n_max)
        operator_dict = compute_tensor_products(n_max, N)
        self.J_x   = operator_dict['J_x']
        self.J_y   = operator_dict['J_y']
        self.J_z   = operator_dict['J_z']
        self.J_m   = operator_dict['J_m']
        self.J_p   = operator_dict['J_p']
        self.a     = operator_dict['a']
        self.a_dag = operator_dict['a_dag']
        self.P     = create_parity_operator()
        self.H     = Dicke_Hamiltonian(N)
        del operator_dict

        # Set variable and find states
        if qs:
            vars = np.linspace(1e-10, 3*λ_critical, 101)
            self.find_states(vars)
            self.sort_states()

    def find_states(self, vars):
        self.vars   = vars
        self.states = calculate_states(vars)

    def sort_states(self, sort_1='P', sort_2='E'):
        self.states, self.quantum_numbers = sort_by_quantum_numbers(self.states, sort=sort_1, secondary_sort=sort_2)

    def select_states(self, set_selection='ground', backup=True, restore=False):
        if restore:
            self.states          = self.states_backup
            self.quantum_numbers = self.quantum_numbers_backup

        elif backup:
            self.states_backup            = self.states
            self.quantum_numbers_backup   = self.quantum_numbers

        self.states, self.quantum_numbers = select_states(self.vars, self.states, self.quantum_numbers, selection=set_selection)

    def plot(self, type):
        if type == 'spectrum':
            plot_list = find_spectrum(self.vars, self.states)
        elif type == 'occupation':
            plot_list = find_occupation(self.vars, self.states)
        elif type == 'squeezing':
            ΔJ_x      = uncertainty(self.states, self.J_x)
            ΔJ_y      = uncertainty(self.states, self.J_y)
            ΔJ_z      = uncertainty(self.states, self.J_z)
            J_x_exp   = expectation(self.J_x, self.states, single_state=False)
            J_x_exp   = tolerance_check(J_x_exp)
            J_y_exp   = expectation(self.J_y, self.states, single_state=False)
            J_z_exp   = expectation(self.J_z, self.states, single_state=False)
            product_1 = ΔJ_x * ΔJ_y
            product_2 = ΔJ_y * ΔJ_z
            product_3 = ΔJ_x * ΔJ_z
            ζ         = bosonic_squeezing(self.states)
            plot_list = [[(f"", f"$ζ^2$"),      (self.vars, ζ),         (0, 1), ('plot')],
                         [(f"", f"$⟨J_x⟩$"),    (self.vars, J_x_exp),   (1, 0), ('plot')],
                         [(f"", f"$⟨J_y⟩$"),    (self.vars, J_y_exp),   (1, 1), ('plot')],
                         [(f"", f"$⟨J_z⟩$"),    (self.vars, J_z_exp),   (1, 2), ('plot')],
                         [(f"", f"$ΔJ_x$"),     (self.vars, ΔJ_x),      (2, 0), ('plot')],
                         [(f"", f"$ΔJ_y$"),     (self.vars, ΔJ_y),      (2, 1), ('plot')],
                         [(f"", f"$ΔJ_z$"),     (self.vars, ΔJ_z),      (2, 2), ('plot')],
                         [(f"", f"$ΔJ_xΔJ_y$"), (self.vars, product_1), (3, 0), ('plot')],
                         [(f"", f"$ΔJ_yΔJ_z$"), (self.vars, product_2), (3, 1), ('plot')],
                         [(f"", f"$ΔJ_xΔJ_z$"), (self.vars, product_3), (3, 2), ('plot')]]
        elif type == 'entropy':
            plot_list = find_entropy(self.vars, self.states, self.n_max, m_J(self.N))
        else:
            print('Try a different keyword.')

        plot_results(plot_list, self.quantum_numbers)

########################################################################################################################################################
# Main
def main():
    examples(9)

if __name__ == "__main__":
    main()

########################################################################################################################################################