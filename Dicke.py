########################################################################################################################################################
#
# Dicke model
# Alex Heinrich
# Sept. 2024
#
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
       3a) Plot ground state expectation values as a function of λ using plot_n_and_Jz()
       3b) Plot all eigenvalues as functions of λ using plot_spectrum()

    Works in progress (WIP)
    -----------------------
    plot_spectrum() : works for ground states and multiple excited states 
    plot_n_and_Jz() : works for ground states, but does not work for excited states
    other           : try x=c(a†+a) with c → λ_default
    other           : convert everything to OOP """

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
    The function quantum_numbers_sorting() can be used to determine initial values for |n, m_J⟩.
    
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
from tqdm import tqdm                            # loading bar

## Computation
import numpy as np                               # tensor algebra
from scipy.special import factorial              # factorials
from scipy.optimize import linear_sum_assignment # eigenvalue reordering

## WIP
import qutip as qt                               # Wigner distribution (deprecated or nonfunctional)
from scipy.linalg import expm                    # unitary transformations (WIP)
from itertools import permutations

########################################################################################################################################################
# Parameters
## Set these here
ω             = 0.1 # field frequency; single-mode field, like first-order waves in a box (10**16 1/s)
ω0            = 10   # atomic frequency; single frequency for the transition |n⟩ → |n ± 1⟩ for a harmonic oscillator (10**6 1/s)
ℏ             = 1   # Planck's constant (1.054571817 * 10**(-34) J∙s)
m             = 1   # mass

## Set these in main()
n_max_default = 3   # number of energy levels, including vacuum state; sets Fock space
N_default     = 3   # number of particles; sets spin space: integer number of m_J for |m_J ,m_J, ..., m_J⟩

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
    H_int   = ℏ / np.sqrt(N) * (a + a_dag) @ J_x   # quantifies the interaction between the atoms and the field
    H       = lambda λ: H_field + H_atom + λ*H_int # sums the total energy and sets the interaction strength

def Dicke_Hamiltonian_bif(N):
    """ Constructs the Hamiltonian given global operator values.
        
        Parameters
        ----------
        N : integer; total number of particles
        
        Features
        --------
        λ : integer; coupling strength """
    
    global H, H_field, H_atom, H_int

    H_field = ℏ * ω  * (a_dag @ a + a @ a)         # counts the energy of each photon; (1/2)*np.eye(a.shape[0])
    H_atom  = ℏ * ω0 * J_z                 # counts the energy of each spin
    H_int   = ℏ / np.sqrt(N) * (a + a_dag) @ J_x   # quantifies the interaction between the atoms and the field
    H       = lambda λ: H_field + H_atom + λ*H_int # sums the total energy and sets the interaction strength

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
        state:       : 1D or 2D array
        single_state : sets whether state is 1D or 2D
        
        Returns
        -------
        expectation_value : float; single_state=True yields one number 
        expectation_array : 2D array; single_state=False has one row per λ """
    
    if single_state:
        expectation_value = state @ operator @ np.conj(state).T
        return expectation_value
    
    else:
        expectation_array = []
        for i in tqdm(range(len(state[1])), desc=f"{'calculating expectation values':<35}"):
            temp_list_1 = []
            for j in range(len(state[1][i][0])):
                temp_list_2 = expectation(operator, state[1][i][:,j])
                temp_list_1.append(temp_list_2)
            expectation_array.append(np.array(temp_list_1).T)
        return np.array(expectation_array)

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

def run_SEOP_Hamiltonian():
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
    H_field = ℏ * ω  * (a_dag @ a)         # counts the energy of each photon; (1/2)*np.eye(a.shape[0])
    H_I     = ℏ * ω0 * I_z                 # counts the energy of each spin
    H_S     = ℏ * ω0 * S_z                 # counts the energy of each spin
    H_spin  = ℏ * ω0 * I_z @ S_z           # counts the energy of each spin
    H_int   = ℏ / np.sqrt(N) * (a + a_dag) @ S_x   # quantifies the interaction between the atoms and the field
    H       = lambda λ: H_field + H_I + H_S + H_spin + λ*H_int # sums the total energy and sets the interaction strength

    # Generate all eigenstates and eigenvalues
    variable_set = np.linspace(0, 5*λ_critical, 101)
    states       = calculate_states(variable_set)

    # Sort eigenstates and eigenvalues
    sort_dict = {'P': P, 'E': H, 'S_z': S_z}
    states, quantum_numbers = quantum_numbers_sorting(states, sort='P', secondary_sort='E')

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
    #plot_spectrum(variable_set, states, quantum_numbers)
    plot_n_S(variable_set, states, quantum_numbers)

def run_2_spin_Hamiltonian():
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
    H_field = ℏ * ω  * (a_dag @ a)         # counts the energy of each photon; (1/2)*np.eye(a.shape[0])
    H_spin  = ℏ * ω0 * (S_z + I_z)                 # counts the energy of each spin
    H_int   = ℏ / np.sqrt(N) * (a + a_dag) @ (S_x - I_x)   # quantifies the interaction between the atoms and the field
    H       = lambda λ: H_field + H_spin + λ*H_int # sums the total energy and sets the interaction strength

    # Generate all eigenstates and eigenvalues
    variable_set = np.linspace(0, 5*λ_critical, 201)
    states       = calculate_states(variable_set)

    # Sort eigenstates and eigenvalues
    sort_dict = {'P': P, 'E': H, 'S_z': S_z, 'n': a_dag@a}
    states, quantum_numbers = quantum_numbers_sorting(states, sort='P', secondary_sort='E')

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
    #plot_spectrum(variable_set, states, quantum_numbers)
    plot_n_S(variable_set, states, quantum_numbers)

def plot_n_and_Jz(variable_set, states, quantum_numbers=None):
    """ Generate and plot ⟨n⟩ and ⟨Jz⟩ for each ground state as a function of λ. """
        
    # Calculate expectation values
    n_expectations   = expectation(a_dag@a, states, single_state=False)
    J_x_expectations = expectation(J_x,     states, single_state=False)
    J_y_expectations = expectation(J_y,     states, single_state=False)
    J_z_expectations = expectation(J_z,     states, single_state=False)

    # Just a weird bug fix
    J_y_expectations = np.real(J_y_expectations)
    
    plot_results([[(f"$λ$", f"$⟨n⟩$"),   (variable_set, n_expectations),   (0, 1), ('plot')],
                  [(f"$λ$", f"$⟨J_x⟩$"), (variable_set, J_x_expectations), (1, 0), ('plot')],
                  [(f"$λ$", f"$⟨J_y⟩$"), (variable_set, J_y_expectations), (1, 1), ('plot')],
                  [(f"$λ$", f"$⟨J_z⟩$"), (variable_set, J_z_expectations), (1, 2), ('plot')]],
                  quantum_numbers = quantum_numbers)

def plot_spectrum(variable_set, states, quantum_numbers=None):
    """ Generate and plot energy eigenvalues.
    
        Optional
        --------
        Subtract the ground state energy from all eigenvalues
            for i in range(len(states[0])):
                for j in range(len(states[0][0])): 
                    val = states[0][i].copy()[0]
                    states[0][i][-(j+1)] -= val """

    # Plot results
    plot_results([[(f"$λ$", f"$E$"), (variable_set, states[0]), (0, 0), ('plot')]],
                   quantum_numbers = quantum_numbers)

def examples(specific_example=0):
    """ Run a preset example.
    
        Parameters
        ----------
        specific_example : natural number
                           0) Introduction; plot a simple spectrum
                           1) Under the hood; plot ⟨n⟩ and ⟨J⟩ for the ground state of each parity
                           2) Phase transition; plot a slightly more complicated spectrum
                           3) Crossing; plot an even more complicated spectrum
                           4) Bifurcations; plot a spectrum with a non-Hermitian Hamiltonian """
    
    global ω, ω0
    
    # Example 0
    if specific_example == 0:
    
        # Set frequencies
        ω, ω0 = 1, 1
    
        # Initialize model
        init_Dicke_model(n_max=2, N=1)

        # Generate all eigenstates and eigenvalues
        variable_set = np.linspace(0, 3*λ_critical, 101)
        states       = calculate_states(variable_set)

        # Sort eigenstates and eigenvalues
        states, quantum_numbers = quantum_numbers_sorting(states, sort='P', secondary_sort='E')

        # Make a calculation
        plot_spectrum(variable_set, states, quantum_numbers)
    
    # Example 1
    elif specific_example == 1:
    
        # Set frequencies
        ω, ω0 = 0.1, 10
        
        # Initialize model
        init_Dicke_model(n_max=24, N=2)

        # Generate all eigenstates and eigenvalues
        variable_set = np.linspace(0, 3*λ_critical, 101)
        states       = calculate_states(variable_set)

        # Sort eigenstates and eigenvalues
        states, quantum_numbers = quantum_numbers_sorting(states, sort='P', secondary_sort='E')

        # Select specific eigenstates
        selected_states = [0, int(len(states[0][0])/2)]
        states          = [states[0][:,selected_states], states[1][:,:,selected_states]]
        quantum_numbers = quantum_numbers[:,selected_states]

        # Make a calculation
        plot_n_and_Jz(variable_set, states, quantum_numbers)
    
    # Example 2
    elif specific_example == 2:
    
        # Set frequencies
        ω, ω0 = 0.1, 10
        
        # Initialize model
        init_Dicke_model(n_max=48, N=4)

        # Generate all eigenstates and eigenvalues
        variable_set = np.linspace(0, 3*λ_critical, 101)
        states       = calculate_states(variable_set)

        # Sort eigenstates and eigenvalues
        states, quantum_numbers = quantum_numbers_sorting(states, sort='P', secondary_sort='E')

        # Make a calculation
        plot_spectrum(variable_set, states, quantum_numbers)
    
    # Example 3
    elif specific_example == 3:
    
        # Set frequencies
        ω, ω0 = 1, 1
        
        # Initialize model
        init_Dicke_model(n_max=48, N=4)

        # Generate all eigenstates and eigenvalues
        variable_set = np.linspace(0, 3*λ_critical, 101)
        states       = calculate_states(variable_set)

        # Sort eigenstates and eigenvalues
        states, quantum_numbers = quantum_numbers_sorting(states, sort='P', secondary_sort='E')

        # Make a calculation
        plot_spectrum(variable_set, states, quantum_numbers)
    
    # Example 4
    elif specific_example == 4:
    
        # Set frequencies
        ω, ω0 = 0.1, 10
        
        # Initialize model
        n_max, N = 48, 4
        create_J_operators(N)              # creates J_p, J_m, J_x, J_y, and J_z given number of particles 
        create_a_operators(n_max)          # creates a and a_dag given number of available energy levels
        compute_tensor_products(n_max, N)  # updates J_p, J_m, J_x, J_y, J_z, a, and a_dag to the full Hilbert space
        create_parity_operator()           # creates parity operator for sorting
        Dicke_Hamiltonian_bif(N)           # uses global operators to construct the full Hamiltonian

        # Generate all eigenstates and eigenvalues
        variable_set = np.linspace(0, 10*λ_critical, 501)
        states       = calculate_states(variable_set)

        # Sort eigenstates and eigenvalues
        states, quantum_numbers = quantum_numbers_sorting(states, sort='n', secondary_sort='J_z')

        # Make a calculation
        plot_spectrum(variable_set, states, quantum_numbers)
    
    else:
        print('There are no examples with this value.')

########################################################################################################################################################
# Utility
def calculate_states(variable_set, ground_state=False):
    """ Computes states in the standard representation.
        Only defined out of convenience, since I run these codes frequently. """
    
    eigenvalues, eigenvectors = [], []
    for λ in tqdm(variable_set, desc=f"{'finding eigenstates':<35}"):
        eigenvalue, eigenvector = eigenstates(H(λ), ground_state)  # Compute once
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)  
    return [np.array(eigenvalues), np.array(eigenvectors)] 

def plot_results(results, quantum_numbers=None):
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

    # Initialize plots after setting grid size
    try:
        max_row = max(item[2][0] for item in results) + 1
        max_col = max(item[2][1] for item in results) + 1
    except:
        max_row = 1
        max_col = 1
    fig = plt.figure(figsize=(3*max_col, 3*max_row))
    gs  = GridSpec(max_row, max_col, figure=fig)
    #plt.style.use('dark_background')
    
    # Define color map
    color_map = {1: 'Reds_r', -1: 'Blues_r'}
    def get_colormap_color(cmap_name, column, num_columns):
        cmap = plt.get_cmap(cmap_name)
        normalized_value = column / num_columns
        capped_value = min(normalized_value, 0.9)
        return cmap(capped_value) # Returns RGBA color
    
    # Sort through each individual plot
    for labels, values, indices, style in tqdm(results, desc=f"{'plotting results':<35}"):
                
        # Initialize the given plot and assign axis labels
        ax = fig.add_subplot(gs[indices[0], indices[1]])
        ax.set_xlabel(labels[0], fontsize=16)
        ax.set_ylabel(labels[1], fontsize=16)

        # Plot with specified coloring
        if quantum_numbers is not None:

            # Sort through y-values
            for i in range(len(values[1][0])):
                color_category = color_map.get(int(quantum_numbers[0][i][0]))
                set_color = get_colormap_color(color_category, i+1, len(values[1][0]))
            
                # Perform plotting based on style
                if style == 'plot':
                    ax.plot(values[0], values[1][:, i], color=set_color)
                elif style == 'contour':
                    ax.contourf(values[0], values[1][:, i], values[2], 100)
        
        # Plot without specified coloring
        else:
            if style == 'plot':
                ax.plot(values[0], values[1])
            elif style == 'contour':
                ax.contourf(values[0], values[1], values[2], 100)
    
    plt.tight_layout()
    plt.show()

def find_quantum_numbers(states, precision=10, sort=None, secondary_sort=None, sort_dict={}):
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
            P_expectations   = [expectation(P,         states[1][i][:, j]) for j in range(len(states[1][0][0]))]
            n_expectations   = [expectation(a_dag @ a, states[1][i][:, j]) for j in range(len(states[1][0][0]))]
            J_z_expectations = [expectation(J_z,       states[1][i][:, j]) for j in range(len(states[1][0][0]))]
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
                    expectations_cache = [expectation(sort_dict[sort], states[1][i][:, j]) for j in range(len(states[1][0][0]))]
                    for k in range(len(states[0][i])):
                        expectations_rounded.append([round(expectations_cache[k], set_precision)])
                    expectation_list.append(np.array(expectations_rounded))
            
            # Calculate two quantum numbers (|sort, secondary_sort⟩)
            else:
                
                # Calculate first number
                if sort == 'E':
                    expectations_cache_1 = states[0][i]
                else:
                    expectations_cache_1 = [expectation(sort_dict[sort], states[1][i][:, j]) for j in range(len(states[1][0][0]))]
                
                # Calculate second number
                if secondary_sort == 'E':
                    expectations_cache_2 = states[0][i]
                else:
                    expectations_cache_2 = [expectation(sort_dict[secondary_sort], states[1][i][:, j]) for j in range(len(states[1][0][0]))]
                for k in range(len(states[0][i])):
                    expectations_rounded.append([round(np.real(expectations_cache_1[k]), set_precision),
                                                 round(np.real(expectations_cache_2[k]), set_precision)])
                expectation_list.append(np.array(expectations_rounded))
    
    return np.array(expectation_list)

def quantum_numbers_sorting(states, sort=None, secondary_sort=None, sort_dict={}):
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
    expectation_list = find_quantum_numbers(states, sort=sort, secondary_sort=secondary_sort, sort_dict=sort_dict)

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

########################################################################################################################################################
# WIP
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

def uncategorized_fn_1_new(states, variable_set):
    ## Prepare entropy arrays
    entropy_tot    = np.zeros(np.shape(variable_set))
    entropy_cavity = np.zeros(np.shape(variable_set))
    entropy_spin   = np.zeros(np.shape(variable_set))
    
    print(states[1])
    print()
    
    ## Calculate entropy for each ground state
    for i, psi_gnd in enumerate(states[1]):
    
        # Compute the density matrix for the ground state
        rho_gnd = np.outer(psi_gnd, psi_gnd.conj())
        
        # Partial trace to get reduced density matrices
        rho_gnd_cavity = partial_trace(rho_gnd, n_max_default, m_J(N_default), trace_out='B')
        rho_gnd_spin   = partial_trace(rho_gnd, n_max_default, m_J(N_default), trace_out='A')
        
        #print(rho_gnd_cavity)
        #print()
        #print(rho_gnd_spin)
        #print()
        
        # Calculate the von Neumann entropy for the total system and subsystems
        entropy_tot[i]    = von_Neumann_entropy(rho_gnd, 2)
        entropy_cavity[i] = von_Neumann_entropy(rho_gnd_spin, 2)
        entropy_spin[i]   = von_Neumann_entropy(rho_gnd_cavity, 2) 
        
        #print(entropy_cavity[i])
        #print()
        #print(entropy_spin[i])
        #print()   
        
    fig, axes = plt.subplots(1, 1, figsize=(12,6))
    axes.plot(variable_set, entropy_tot, 'k', variable_set, entropy_cavity, 'b', variable_set, entropy_spin, 'r--')

    axes.set_ylabel("Entropy of subsystems", fontsize=16)
    axes.set_xlabel("interaction strength", fontsize=16)

    plt.tight_layout()
    plt.show()

def uncategorized_fn_1():
    ##
    psi_gnd_sublist = ground_state_list[1][::4]
    xvec            = np.linspace(-7,7,200)

    ##
    entropy_tot    = np.zeros(np.shape(λ_vals))
    entropy_cavity = np.zeros(np.shape(λ_vals))
    entropy_spin   = np.zeros(np.shape(λ_vals))
    for idx, psi_gnd in enumerate(ground_state_list[1]):
        rho_gnd_cavity = qt.ptrace(psi_gnd, 0)
        rho_gnd_spin   = qt.ptrace(psi_gnd, 1)
        entropy_tot[idx]    = qt.entropy_vn(psi_gnd, 2)
        entropy_cavity[idx] = qt.entropy_vn(rho_gnd_cavity, 2)
        entropy_spin[idx]   = qt.entropy_vn(rho_gnd_spin, 2)
    
    fig, axes = plt.subplots(1, 1, figsize=(12,6))
    axes.plot(g_vec, entropy_tot, 'k', g_vec, entropy_cavity, 'b', g_vec, entropy_spin, 'r--')

    axes.set_ylim(0, 1.5)
    axes.set_ylabel("Entropy of subsystems", fontsize=16)
    axes.set_xlabel("interaction strength", fontsize=16)

    fig.tight_layout()

def calulcate_entropy(n_max, N, λ_vals):
    J = N/2.0
    m_J = 2*J + 1

    # setup the hamiltonian for the requested hilbert space sizes
    a  = qt.tensor(qt.destroy(n_max), qt.qeye(m_J))
    Jp = qt.tensor(qt.qeye(n_max), qt.jmat(J, '+'))
    Jm = qt.tensor(qt.qeye(n_max), qt.jmat(J, '-'))
    Jz = qt.tensor(qt.qeye(n_max), qt.jmat(J, 'z'))

    H0 = ω * a.dag() * a + ω0 * Jz
    H1 = 1.0 / np.sqrt(N) * (a + a.dag()) * (Jp + Jm)

    # Ground state and steady state for the Hamiltonian: H = H0 + g * H1
    psi_gnd_list = [(H0 + g * H1).groundstate()[1]  for g in λ_vals]
    
    entropy_cavity = np.zeros(np.shape(λ_vals))
    entropy_spin   = np.zeros(np.shape(λ_vals))

    for idx, psi_gnd in enumerate(psi_gnd_list):

        rho_gnd_cavity = qt.ptrace(psi_gnd, 0)
        rho_gnd_spin   = qt.ptrace(psi_gnd, 1)
    
        entropy_cavity[idx] = qt.entropy_vn(rho_gnd_cavity, 2)
        entropy_spin[idx]   = qt.entropy_vn(rho_gnd_spin, 2)
        
    return entropy_cavity, entropy_spin

def uncategorized_fn_2():
    ##
    # average number thermal photons in the bath coupling to the resonator
    n_th = 0.25

    c_ops = [np.sqrt(κ * (n_th + 1)) * a, np.sqrt(κ * n_th) * a.dag()]
    #c_ops = [sqrt(κ) * a, sqrt(gamma) * Jm]
    g_vec = np.linspace(0.01, 1.0, 20)

    # Ground state for the Hamiltonian: H = H0 + g * H1
    rho_ss_list = [qt.steadystate(H_field + H_atom + g * H_int, c_ops) for g in g_vec]
    # calculate the expectation value of the number of photons in the cavity
    n_ss_vec = qt.expect(a.dag() * a, rho_ss_list)

def plot_entropy_entanglement():
    λ_vals = np.linspace(0.2, 0.8, 60)
    N_vec = [4, 8, 12, 16, 24, 32]
    MM = 25

    fig, axes = plt.subplots(1, 1, figsize=(12,6))

    for NN in N_vec:
        
        entropy_cavity, entropy_spin = calulcate_entropy(MM, NN, λ_vals)
        
        axes.plot(λ_vals, entropy_cavity, 'b', label="N = %d" % NN)
        axes.plot(λ_vals, entropy_spin, 'r--')

    axes.set_ylim(0, 1.75)
    axes.set_ylabel("Entropy of subsystems", fontsize=16)
    axes.set_xlabel("interaction strength", fontsize=16)
    axes.legend()

    fig.tight_layout()
    plt.show()

def plot_entropy_subsystems():
    entropy_tot    = np.zeros(np.shape(λ_vals))
    entropy_cavity = np.zeros(np.shape(λ_vals))
    entropy_spin   = np.zeros(np.shape(λ_vals))

    for idx, rho_ss in enumerate(rho_ss_list):

        rho_gnd_cavity = qt.ptrace(rho_ss, 0)
        rho_gnd_spin   = qt.ptrace(rho_ss, 1)
        
        entropy_tot[idx]    = qt.entropy_vn(rho_ss, 2)
        entropy_cavity[idx] = qt.entropy_vn(rho_gnd_cavity, 2)
        entropy_spin[idx]   = qt.entropy_vn(rho_gnd_spin, 2)
         

    fig, axes = plt.subplots(1, 1, figsize=(12,6))

    axes.plot(λ_vals, entropy_tot, 'k', label="total")
    axes.plot(λ_vals, entropy_cavity, 'b', label="cavity")
    axes.plot(λ_vals, entropy_spin, 'r--', label="spin")

    axes.set_ylabel("Entropy of subsystems", fontsize=16)
    axes.set_xlabel("interaction strength", fontsize=16)
    axes.legend(loc=0)
    fig.tight_layout()
    plt.show()

def plot_Wigner_distribution_1(variable_set, states):
    """ Uses QuTiP to plot Wigner distribution. """
    
    # Select some states to plot
    ground_state_sublist = states[1][::4]

    # Generate phase space intervals
    xvec = np.linspace(-7, 7, 200) # np.linspace(np.min(x_eigenvalues), np.max(x_eigenvalues), 100)
    pvec = xvec # np.linspace(np.min(p_eigenvalues), np.max(p_eigenvalues), 100)
    
    # Calculate Wigner distributions
    plot_list = []
    for i in tqdm(range(len(ground_state_sublist)), desc=f"{'creating Wigner distributions':<35}"):
    
        # Calculate density matrix
        ρ = np.outer(ground_state_sublist[i], ground_state_sublist[i].conj())
        
        # Trace out the spin space
        ρ = partial_trace(ρ, n_max_default, int(2*J(N_default)+1), trace_out='B')
        
        # Calculate using QuTiP
        W = qt.wigner(qt.Qobj(ρ), xvec, pvec)
        
        # Return for plotting
        plot_list.append([(f"$x$", f"$p$"), (xvec, pvec, W), (0, i), ('contour')])

    plot_results(plot_list)

def plot_Wigner_distribution_2(rho, xvec, pvec):
    """
    Calculate the Wigner function for a quantum state in the Fock basis.

    Parameters:
    rho : ndarray
        Density matrix in the Fock basis.
    xvec, pvec : ndarray
        Arrays representing the phase space grid.

    Returns:
    W : ndarray
        2D array of the Wigner function values over the phase space grid.
    """
    N = rho.shape[0]  # Dimension of the Hilbert space
    W = np.zeros((len(xvec), len(pvec)), dtype=np.complex128)
    
    # Create a grid of alpha = (x + ip) / sqrt(2)
    for i, x in tqdm(enumerate(xvec), desc=f"{'calculating Wigner distribution':<35}"):
        for j, p in enumerate(pvec):
            alpha = (x + 1j * p) / np.sqrt(2)
            
            # Displacement operator in the Fock basis
            D = np.zeros((N, N), dtype=np.complex128)
            for m in range(N):
                for n in range(N):
                    if m >= n:
                        D[m, n] = np.exp(-0.5 * np.abs(alpha)**2) * (alpha ** (m-n)) / np.sqrt(factorial(m) * factorial(n))
                    else:
                        D[m, n] = 0
            
            # Calculate Wigner function at point (x, p)
            W[i, j] = np.trace(np.dot(rho, D)) * np.exp(-np.abs(alpha)**2)
    
    W = np.real(W)  # Wigner function should be real
    W /= np.pi  # Normalization

    return W

def Wigner_function_optimized(rho, xvec, pvec):
    Nx = len(xvec)
    W = np.zeros((Nx, Nx), dtype=np.complex128)

    # Create grids for phase space
    X, P = np.meshgrid(xvec, pvec, indexing='ij')
    alpha_grid = (X + 1j * P) / np.sqrt(2)

    # Calculate Wigner function using FFT
    for i in tqdm(range(Nx), desc='calculating Wigner distribution'):
        for j in range(Nx):
            alpha = alpha_grid[i, j]
            D = np.zeros(rho.shape, dtype=np.complex128)
            for m in range(rho.shape[0]):
                for n in range(rho.shape[1]):
                    if m >= n:
                        D[m, n] = np.exp(-0.5 * np.abs(alpha)**2) * (alpha ** (m-n)) / np.sqrt(factorial(m) * factorial(n))
            W[i, j] = np.trace(np.dot(rho, D)) * np.exp(-np.abs(alpha)**2)

    W = np.real(W) / np.pi
    return W

def time_evolution(state):
    variable_set  = np.linspace(0, 10, 11)
    time_operator = lambda t: expm(-1j*H(0)*t/ℏ)
    E_vals = [expectation(H(0), time_operator(t)@state) for t in variable_set]
    
    plot_results([[(f"$t$", f"$E$"), (variable_set, E_vals), (0, 0), ('plot')]])

def tensor_product_to_vectors(tensor, dim_1, dim_2):
    # Reconstruct the first vector (space 1)
    vector_1 = np.zeros(dim_1)
    for i in range(dim_1):
        vector_1[i] = tensor_product_vector[i * dim_2] / tensor_product_vector[0]  # Normalize by first element

    # Reconstruct the second vector (space 2)
    vector_2 = np.zeros(dim_2)
    for j in range(dim_2):
        vector_2[j] = tensor_product_vector[j] / vector_1[0]  # Use the first vector to reconstruct the second

    return vector_1, vector_2

def create_x_p_operators():
    """ Possibly useful for Wigner distribution plots. """
    
    global x, p
    
    try:
        x = np.sqrt(ℏ/(2*m*ω)) * (a_dag + a)
        p = 1j * np.sqrt(ℏ*m*ω/2) * (a_dag - a)
    except:
        create_a_operators(n_max_default)
        x = np.sqrt(ℏ/(2*m*ω)) * (a_dag + a)
        p = 1j * np.sqrt(ℏ*m*ω/2) * (a_dag - a)

########################################################################################################################################################
# Main
def main():
    """ Ideally only used to develop algorithms. """
    
    #examples(1)
    #run_SEOP_Hamiltonian()
    run_2_spin_Hamiltonian()
    
    # Initialize model
    #init_Dicke_model(n_max_default, N_default)

    # Generate all eigenstates and eigenvalues
    #variable_set = np.linspace(0, 1.0, 2) # np.linspace(0, 3*λ_critical, 101) 
    #states = calculate_states(variable_set, ground_state=False)

    # Sort eigenstates and eigenvalues
    #states, quantum_numbers = quantum_numbers_sorting(states, sort='P', secondary_sort='E')

    # Do some optional stuff
    ## Verify sorting (optional)
    #print(find_quantum_numbers(states, precision=1, sort=None, secondary_sort=None)[0])
    #print(find_quantum_numbers(states, precision=1, sort=None, secondary_sort=None)[-1])
    ## Select specific eigenstates; indexed from least to greatest energy
    #selected_states = [0] # ex. range(15) or [0, len(states[0])/2]
    #states          = [states[0][:,selected_states], states[1][:,:,selected_states]]
    #quantum_numbers = quantum_numbers[:,selected_states]

    # Make a calculation
    #plot_spectrum(variable_set, states, quantum_numbers=None)
    #plot_n_and_Jz(variable_set, states, quantum_numbers=None)
    #uncategorized_fn_1_new(states, variable_set)

if __name__ == "__main__":
    main()

########################################################################################################################################################