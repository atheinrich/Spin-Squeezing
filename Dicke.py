########################################################################################################################################################
# Dicke model
########################################################################################################################################################

########################################################################################################################################################
# Summary
""" Overview
    
    Dicke model
    -----------
    My current understanding is that a†a counts the number of photons absorbed by the atoms from the field,
    and J_z counts the spin energy of the atoms in the z-direction. Without the interaction term,
    the atoms are in the typical ground state, such that they have no excitations. For nonzero coupling
    strengths, the atoms are able to absorb and emit photons with equal probability (?).
    The total spin should be conserved, but that is not apparent in the numerical models to me.

    Command line example
    --------------------
    sys = System(qs=True) # generate system with default parameters
    sys.print()           # view parameters
    sys.plot('spectrum')  # plot the full spectrum
    sys.N = 2             # change the number of atoms
    sys.update()          # update the system to account for the new number of atoms
    sys.plot('spectrum')  # plot the full spectrum
    sys.save()            # save data
    sys = examples(0)     # update the system to a preset example
    sys = sys.load()      # load the previous system """

""" Organization

    Imports    : packages used to streamline calculations and data visualization
    Parameters : global values not set elsewhere; mostly defaults, except for J and m_J
    Operators  : construction of global operators
    Operations : actions made on operators and state vectors
    Algorithms : actions made on operators and state vectors    
    Utility    : processes that are not specific to the model being simulated
    WIP        : functions and algorithms that should not be relied on but have potential value
    Main       : ideally only used for algorithm development """

""" Data descriptions

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
import pickle                                    # file saving

## Computation
import numpy as np                               # tensor algebra
from scipy.linalg import expm                    # unitary transformations

########################################################################################################################################################
# Operators
def create_J_operators(sys, j_set=None, individual=False):
    """ Generates total angular momentum matrices in the Fock basis.
        
        Parameters
        ----------
        N   : integer; total number of particles
        
        Returns
        -------
        J_p : matrix; raising operator for collective angular momentum
        J_m : matrix; lowering operator for collective angular momentum
        J_x : matrix; x-component operator for collective angular momentum
        J_y : matrix; y-component operator for collective angular momentum
        J_z : matrix; z-component operator for collective angular momentum """
    
    # Construct operators in full space, rather than collective space
    if individual:
        identity = np.eye(2)

        dim = 2**sys.N
        J_x = np.zeros((dim, dim), dtype=complex)
        J_y = np.zeros((dim, dim), dtype=complex)
        J_z = np.zeros((dim, dim), dtype=complex)
        
        for i in range(sys.N):
            J_x_cache = 1
            J_y_cache = 1
            J_z_cache = 1
            
            for j in range(sys.N):
                if j == i:
                    J_x_cache = np.kron(J_x_cache, (sys.ℏ/2) * np.array([[0, 1], [1, 0]]))
                    J_y_cache = np.kron(J_y_cache, (sys.ℏ/2) * np.array([[0, -1j], [1j, 0]]))
                    J_z_cache = np.kron(J_z_cache, (sys.ℏ/2) * np.array([[1, 0], [0, -1]]))
                else:
                    J_x_cache = np.kron(J_x_cache, np.eye(2))
                    J_y_cache = np.kron(J_y_cache, np.eye(2))
                    J_z_cache = np.kron(J_z_cache, np.eye(2))
            
            # Add to the sum
            J_x += J_x_cache
            J_y += J_y_cache
            J_z += J_z_cache
        J_m = (J_x - 1j * J_y) / 2
        J_p = (J_x + 1j * J_y) / 2
    
        #J_x = np.array([[0, 1], [1, 0]])/2
        #for i in range(sys.N-1):
        #    J_x = np.kron(J_x, np.array([[0, 1], [1, 0]])/2)
        #J_y = np.array([[0, -1j], [1j, 0]])/2
        #for i in range(sys.N-1):
        #    J_y = np.kron(J_y, np.array([[0, -1j], [1j, 0]])/2)
        #J_z = np.array([[0, 1],   [1, 0]])/2
        #for i in range(sys.N-1):
        #    J_z = np.kron(J_z, np.array([[1, 0], [0, -1]])/2)
        #J_m = (J_x - 1j * J_y) / 2
        #J_p = (J_x + 1j * J_y) / 2
        
        return {'J_x': J_x, 'J_y': J_y, 'J_z': J_z, 'J_m': J_m, 'J_p': J_p}
        
    # Allows for manual setting of j and m_j
    if j_set:
        j         = sys.N * j_set
        dimension = int(round(2 * j_set + 1))
    else:
        j         = sys.J
        dimension = sys.m_J

    if sys.N == 0:
        J = np.eye(dimension)
        return {'J_x': J, 'J_y': J, 'J_z': J, 'J_m': J, 'J_p': J}

    # Ladder operators
    J_p = np.zeros((dimension, dimension))
    J_m = np.zeros((dimension, dimension))
    for i in tqdm(range(dimension - 1), desc=f"{'creating J operators':<35}"):
        m           = j - i
        J_p[i, i+1] = sys.ℏ * np.sqrt(j*(j+1)-m*(m-1))
        J_m[i+1, i] = sys.ℏ * np.sqrt(j*(j+1)-m*(m-1))

    # Component operators
    J_x = (1/2) *(J_p + J_m)
    J_y = (1/2j)*(J_p - J_m)
    J_z = sys.ℏ * np.diag([j-m for m in range(dimension)])
    
    return {'J_x': J_x, 'J_y': J_y, 'J_z': J_z, 'J_m': J_m, 'J_p': J_p}

def create_a_operators(sys):
    """ Generates creation and annihilation matrices in the Fock basis.
        
        Parameters
        ----------
        n_max : integer; number of excitations allowed per atom
        
        Returns
        -------
        a     : matrix; creation operator for photon field
        a_dag : matrix; annihilation operator for photon field """
        
    a = np.zeros((sys.n_max, sys.n_max))
    for i in tqdm(range(1, sys.n_max), desc=f"{'creating a operators':<35}"):
        a[i-1, i] = np.sqrt(i)
    a_dag = a.conj().T
    return {'a': a, 'a_dag': a_dag}
    
def create_parity_operator(sys):
    """ Generate parity operator that commutes with the standard Dicke model. """
    
    field_parity = expm(1j * np.pi * sys.a_dag_field @ sys.a_field)
    spin_parity  = expm(1j * np.pi * sys.J_z_spin)
    
    P = np.real(np.kron(field_parity, spin_parity))
    return tolerance_check(P)

def compute_tensor_products(sys):
    """ Takes the tensor product of the field and atom operators and yields the full Hamiltonian.
        
        Parameters
        ----------
        n_max : integer; number of excitations allowed per atom
        N     : integer; total number of particles """
    
    a     = np.kron(sys.a_field,     np.eye(sys.m_J))
    a_dag = np.kron(sys.a_dag_field, np.eye(sys.m_J))
    
    J_p   = np.kron(np.eye(sys.n_max), sys.J_p_spin) # raises total m_j value, but does not alter the number of photons
    J_m   = np.kron(np.eye(sys.n_max), sys.J_m_spin) # lowers total m_j value, but does not alter the number of photons
    J_x   = np.kron(np.eye(sys.n_max), sys.J_x_spin) 
    J_y   = np.kron(np.eye(sys.n_max), sys.J_y_spin) 
    J_z   = np.kron(np.eye(sys.n_max), sys.J_z_spin) # yields the total m_j value, but does not alter the number of photons

    return {'J_x': J_x, 'J_y': J_y, 'J_z': J_z, 'J_m': J_m, 'J_p': J_p, 'a': a, 'a_dag': a_dag}

########################################################################################################################################################
# Operations
def eigenstates(matrix):
    """ Calculates eigenvalues and eigenvectors for the given matrix.
        For some reason, eigh provides the same eigenvectors as QuTiP, but eig does not.
        
        Parameters
        ----------
        matrix                      : 2D array

        Returns
        -------
        [eigenvalues, eigenvectors] : list of arrays; eigenvalues and eigenvectors
                                      ex. [array(list_of_eigenvalues), array(eigenvectors)] """
    
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return [eigenvalues, eigenvectors]

def expectation(operator, state, single_state=True):
    """ Just shorthand for some numpy methods. 
        
        Parameters
        ----------
        operator          : 2D array
        state:            : standard or column vector
        single_state      : flags as column vector
        
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
                temp_list_2 = expectation(operator, row_to_column(state[1][i][:,j]))
                temp_list_1.append(temp_list_2)
            expectation_array.append(np.array(temp_list_1).T)
        
        return np.array(expectation_array)

def uncertainty(operator, states):
    """ Calculates the standard deviation of a given operator and a set of states. """

    # Initialize data containers
    expectations, output = [[], []], []
    
    # Calculate expectation values
    expectations[0] = expectation(operator,            states, single_state=False)
    expectations[1] = expectation(operator @ operator, states, single_state=False)
    
    # Use expectation values to calculate uncertainty
    for i in range(len(expectations[0])):
        cache = []
        for j in range(len(list(expectations[0][i]))):
            cache.append(np.sqrt(abs(list(expectations[1])[i][j]-list(expectations[0])[i][j]**2)))
        output.append(cache)
    return np.array(output)

def partial_trace(ρ, dim_A, dim_B, subsystem):
    """ Computes the partial trace of a matrix.

        Parameters
        ----------
        ρ         : 2D array; density matrix 
        dim_A     : integer; dimension of subsystem A
        dim_B     : integer; dimension of subsystem B
        subsystem : string in {'A', 'B'}; subsystem to be traced out

        Returns
        -------
        ρ_reduced : reduced matrix after the partial trace """
    
    ρ = ρ.reshape((dim_A, dim_B, dim_A, dim_B))
    if subsystem == 'B':
        ρ_reduced = np.trace(ρ, axis1=1, axis2=3)
    elif subsystem == 'A':
        ρ_reduced = np.trace(ρ, axis1=0, axis2=2)
    return ρ_reduced

def partial_transpose(ρ, dim_A, dim_B, subsystem):
    """ Perform the partial transposition of a bipartite density matrix.
    
        Parameters
        ----------
        ρ         : 2D array; density matrix 
        dim_A     : integer; dimension of subsystem A
        dim_B     : integer; dimension of subsystem B
        subsystem : string in {'A', 'B'}; subsystem to be traced out

        Returns
        -------
        eigensum  : float; sum of the absolute value of each eigenvalue """
    
    # Reshape ρ to handle subsystems separately
    ρ = ρ.reshape((dim_A, dim_B, dim_A, dim_B))
    
    # Perform partial transposition
    if subsystem == 'A':
        ρ = np.transpose(ρ, (2, 1, 0, 3))
    elif subsystem == 'B':
        ρ = np.transpose(ρ, (0, 3, 2, 1))
    
    # Return to original shape
    ρ = ρ.reshape((dim_A * dim_B, dim_A * dim_B))
    
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

########################################################################################################################################################
# Algorithms
def find_occupation(sys):
    """ Prepare ⟨n⟩ and ⟨J_z⟩ for plotting. """
    
    # Calculate expectation values
    n_expectations    = expectation(sys.a_dag @ sys.a, sys.states, single_state=False)
    J_z_expectations  = expectation(sys.J_z,           sys.states, single_state=False)
    
    # Construct and return plot list
    plot_list = [[(f"$λ$", f"$⟨n⟩$"),   (sys.vars, n_expectations),   (0, 0), ('plot')],
                 [(f"$λ$", f"$⟨J_z⟩$"), (sys.vars, J_z_expectations), (0, 1), ('plot')]]
    return plot_list

def find_spectrum(sys):
    """ Prepare energy eigenvalues for plotting.
    
        Optional
        --------
        Subtract the ground state energy from all eigenvalues
            for i in range(len(states[0])):
                for j in range(len(states[0][0])): 
                    val = states[0][i].copy()[0]
                    states[0][i][-(j+1)] -= val """

    return [[(f"$λ$", f"$E$"), (sys.vars, sys.states[0]), (0, 0), ('plot')]]

def find_entropy(sys):
    """ Computes von Neumann entropy and partial transposition for plotting. """

    def von_Neumann_entropy(ρ, base=2):
        """ Calculate the von Neumann entropy via S(ρ) = -Tr(ρ log ρ).
        
            Parameters
            ----------
            ρ       : 2D array; density matrix
            base    : float; base of the logarithm
            
            Returns
            -------
            entropy : float; von Neumann entropy """
        
        eigvals = np.linalg.eigvalsh(ρ)
        eigvals = eigvals[eigvals > 0]                   # avoids issues with log(0)
        
        if base == 2:
            log_eigvals = np.log2(eigvals)
        elif base == np.e:
            log_eigvals = np.log(eigvals)
        else:
            log_eigvals = np.log(eigvals) / np.log(base)
        
        return -np.sum(eigvals * log_eigvals)

    # Prepare entropy arrays
    entropy_tot   = np.zeros_like(sys.states[0])
    entropy_field = np.zeros_like(sys.states[0])
    entropy_spin  = np.zeros_like(sys.states[0])
    eigensum_spin = np.zeros_like(sys.states[0])
    
    # Sort through trials
    for i in tqdm(range(len(sys.states[1])), desc=f"{'calculating entropy':<35}"):
    
        # Sort through states:
        for j in range(len(sys.states[1][0][0])):
            
            # Extract state and compute density matrix
            state = row_to_column(sys.states[1][i][:,j])
            ρ     = np.outer(state, state.conj())
            
            # Compute reduced density matrices
            ρ_spin  = partial_trace(ρ, sys.n_max, sys.m_J, subsystem='A')
            
            # Compute bipartite negativity
            eigensum = partial_transpose(ρ, sys.n_max, sys.m_J, subsystem='A')
            
            # Calculate the von Neumann entropy for the total system and subsystems
            entropy_field[i][j] = von_Neumann_entropy(ρ_spin,  base=2)
            eigensum_spin[i][j] = (eigensum - 1)/2
    
    plot_list = [[(f"$λ$", f"von Neumann entropy"), (sys.vars, entropy_field), (0, 0), ('plot')],
                 [(f"$λ$", f"partial transpose"),   (sys.vars, eigensum_spin), (0, 1), ('plot')]]
    return plot_list

def find_squeezing(sys):
    """ Calculates the standard deviation of a given operator and a set of states. """

    # Initialize data containers
    expectations, output = [[], [], []], []
    
    # Calculate expectation values
    expectations[0] = expectation(sys.a_dag @ sys.a, sys.states, single_state=False)
    expectations[1] = expectation(sys.a,             sys.states, single_state=False)
    expectations[2] = expectation(sys.a @ sys.a,     sys.states, single_state=False)
    
    # Use expectation values to calculate uncertainty
    for i in range(len(expectations[0])):
        cache = []
        for j in range(len(list(expectations[0][i]))):
            factor = 1 + 2*(expectations[0][i][j] - abs(expectations[1][i][j])**2 - abs(expectations[2][i][j] - expectations[1][i][j]**2))
            cache.append(factor)
        output.append(cache)
    return np.array(output)

def plot_data(sys, selection):
    """ Generates data to be sent to plot_handling(). """
    
    sys.print('parameters')
    sys.print('numbers')
    
    if selection == 'spectrum':
        plot_list = find_spectrum(sys)
        
    elif selection == 'occupation':
        plot_list = find_occupation(sys)
        
    elif selection == 'squeezing':
        ΔJ_x      = uncertainty(sys.J_x, sys.states)
        ΔJ_y      = uncertainty(sys.J_y, sys.states)
        ΔJ_z      = uncertainty(sys.J_z, sys.states)
        J_x_exp   = expectation(sys.J_x, sys.states, single_state=False)
        J_x_exp   = tolerance_check(J_x_exp)
        J_y_exp   = expectation(sys.J_y, sys.states, single_state=False)
        J_z_exp   = expectation(sys.J_z, sys.states, single_state=False)
        product_1 = ΔJ_x * ΔJ_y
        product_2 = ΔJ_y * ΔJ_z
        product_3 = ΔJ_x * ΔJ_z
        ζ         = find_squeezing(sys)
        plot_list = [[(f"", f"$ζ^2$"),      (sys.vars, ζ),         (0, 1), ('plot')],
                     [(f"", f"$⟨J_x⟩$"),     (sys.vars, J_x_exp),   (1, 0), ('plot')],
                     [(f"", f"$⟨J_y⟩$"),     (sys.vars, J_y_exp),   (1, 1), ('plot')],
                     [(f"", f"$⟨J_z⟩$"),     (sys.vars, J_z_exp),   (1, 2), ('plot')],
                     [(f"", f"$ΔJ_x$"),     (sys.vars, ΔJ_x),      (2, 0), ('plot')],
                     [(f"", f"$ΔJ_y$"),     (sys.vars, ΔJ_y),      (2, 1), ('plot')],
                     [(f"", f"$ΔJ_z$"),     (sys.vars, ΔJ_z),      (2, 2), ('plot')],
                     [(f"", f"$ΔJ_xΔJ_y$"), (sys.vars, product_1), (3, 0), ('plot')],
                     [(f"", f"$ΔJ_yΔJ_z$"), (sys.vars, product_2), (3, 1), ('plot')],
                     [(f"", f"$ΔJ_xΔJ_z$"), (sys.vars, product_3), (3, 2), ('plot')]]
    
    elif selection == 'entropy':
        plot_list = find_entropy(sys)
    
    elif selection == 'all':
        plot_list_E       = find_spectrum(sys)
        plot_list_E.append([(f"λ", f"$ζ^2$"), (sys.vars, find_squeezing(sys)), (0, 1), ('plot')])
        plot_list_n_J     = find_occupation(sys)
        plot_list_entropy = find_entropy(sys)
        
        # Save for later
        sys.plot_lists = [[plot_list_E, plot_list_n_J, plot_list_entropy], sys.quantum_numbers]
        
        plot_handling(plot_list_E,       sys.quantum_numbers, no_show=True)
        plot_handling(plot_list_n_J,     sys.quantum_numbers, no_show=True)
        plot_handling(plot_list_entropy, sys.quantum_numbers, no_show=True)
        plt.show()
    
    elif selection == 'last':
        for i in range(len(sys.plot_lists[0])):
            plot_handling(sys.plot_lists[0][i], sys.plot_lists[1], no_show=True)
        plt.show()
    
    else:
        print('Try a different keyword.')

    # Look for plots whose handling was done above
    if selection not in ['all', 'last']:
        
        # Save for later
        sys.plot_lists = [[plot_list], sys.quantum_numbers]
        
        # Plot
        plot_handling(plot_list, sys.quantum_numbers)    

def examples(specific_example):
    """ Run a preset example.
        
        Parameters
        ----------
        specific_example    : nonnegative integer
           0 (Tutorial 0)   : the most basic options
           1 (Tutorial 1)   : custom parameters and variables
           2 (Tutorial 2)   : custom state selection
           3 (Tutorial 3)   : putting it all together
           4 (Entropy)      : plot entropy
           4 (bifurcations) : plot a find_spectrum with a non-Hermitian Hamiltonian """
        
    # Tutorial 0: introduction
    if specific_example == 0:
    
        # Initialize model and generate eigenstates
        sys = System(qs=True)

        # Sort eigenstates and eigenvalues
        sys.sort()

        # Make a calculation
        sys.plot('spectrum')
        
        # Return for use in command line
        return sys

    # Tutorial 1: setting parameters
    elif specific_example == 1:
    
        # Initialize model
        field_frequency       = 1
        atomic_frequency      = 1
        number_of_field_modes = 48
        number_of_atoms       = 4
        sys                   = System(field_frequency, atomic_frequency,
                                       number_of_field_modes, number_of_atoms)

        # Generate eigenstates and eigenvalues
        vars = np.linspace(1e-10, sys.crit, 101)
        sys.Hamiltonian(vars)

        # Sort eigenstates and eigenvalues
        sys.sort()

        # Make a calculation
        sys.plot('spectrum')
        
        # Return for use in command line
        return sys
    
    # Tutorial 2: selecting states
    elif specific_example == 2:
    
        # Initialize model
        sys = System(0.1, 10, 24, 2)

        # Generate eigenstates and eigenvalues
        sys.Hamiltonian(np.linspace(1e-10, 2*sys.crit, 101))
        
        # Sort eigenstates and eigenvalues
        sys.sort()

        # Select specific eigenstates
        sys.select(set_selection='ground')

        # Make a calculation
        sys.plot('occupation')
        
        # Return for use in command line
        return sys

    # Tutorial 3: best practice
    elif specific_example == 3:
    
        # Initialize model
        sys = System(0.1, 10, 48, 4)

        # Generate all eigenstates and eigenvalues
        sys.Hamiltonian(np.linspace(1e-10, 3*sys.crit, 101))

        # Sort eigenstates and eigenvalues
        sys.sort('P', 'E')
        
        # Select specific eigenstates
        sys.select([0, 1, 2])
        
        # View parameters
        sys.print()
        
        # Make a calculation
        sys.plot('squeezing')
        
        # Return for use in command line
        return sys

    # Entropy
    elif specific_example == 4:
    
        # Initialize model
        sys = System(0.1, 10, 48, 4)

        # Generate all eigenstates and eigenvalues
        sys.Hamiltonian(np.linspace(0, 6*sys.crit, 101))

        # Sort eigenstates and eigenvalues
        sys.sort('P', 'E')

        # Select specific eigenstates
        sys.select([0, 1 ,2])
        
        # View parameters
        sys.print()
        
        # Make a calculation
        sys.plot('entropy')
        
        # Return for use in command line
        return sys

    # Development: bifurcations
    elif specific_example == 5:
    
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
        H_field = sys.ℏ * ω  * (a_dag @ a + a @ a)
        H_atom  = sys.ℏ * ω0 * J_z
        H_int   = 2 * sys.ℏ / np.sqrt(N) * (a + a_dag) @ J_x
        H       = lambda λ: H_field + H_atom + λ*H_int

        # Generate eigenstates and eigenvalues
        vars = np.linspace(1e-10, 10*λ_critical, 101)
        states       = calculate_states(vars)

        # Sort eigenstates and eigenvalues
        states, quantum_numbers = sort_by_quantum_numbers(states, sort='P', secondary_sort='E')

        # Make a calculation
        plot_list = find_spectrum(vars, states)
        plot_handling(plot_list, quantum_numbers)
    
    # Development: Chebyshev evolution
    elif specific_example == 6:
    
        # Set parameters
        ω, ω0      = 0.1, 10
        n_max, N   = 12,  2
        λ_critical = (ω * ω0)**(1/2)/2
        print_parameters(ω, ω0, n_max, N)
        
        # Initialize model
        init_Dicke_model(n_max, N)

        # Generate eigenstates and eigenvalues
        vars = np.linspace(1e-10, 2*λ_critical, 11)
        states       = calculate_states(vars)

        # Sort eigenstates and eigenvalues
        states, quantum_numbers = sort_by_quantum_numbers(states, sort='P', secondary_sort='E')

        # Select specific eigenstates
        states, quantum_numbers = select_states(vars, states, quantum_numbers, selection="random")

        # Make a calculation
        plot_list = Chebyshift(vars, states)
        plot_handling(plot_list, quantum_numbers, plot_mode="3D")

    # Development: Lindbladian evolution
    elif specific_example == 7:
    
        # Set parameters
        ω, ω0    = 0.1, 10
        n_max, N = 24,  2
        λ_critical = (ω * ω0)**(1/2)/2
        print_parameters(ω, ω0, n_max, N)
        
        # Initialize model
        init_Dicke_model(n_max, N)

        # Generate eigenstates and eigenvalues
        vars = np.linspace(1e-10, 2*λ_critical, 11)
        states       = calculate_states(vars)

        # Sort eigenstates and eigenvalues
        states, quantum_numbers = sort_by_quantum_numbers(states, sort='P', secondary_sort='E')

        # Select specific eigenstates
        states, quantum_numbers = select_states(vars, states, quantum_numbers, selection="ground")

        # Make a calculation
        plot_list = Lindbladian(vars, states)
        plot_handling(plot_list, quantum_numbers, plot_mode="3D")

    # Development: SEOP
    elif specific_example == 8:
        SEOP_Dicke_model()

    else:
        print('There are no examples with this value.')

########################################################################################################################################################
# Utility
def row_to_column(array):
    return array.reshape(array.shape[0], 1)

def calculate_states(sys):
    """ Computes states in the standard representation.
        See data descriptions in Summary for more details. """
        
    eigenvalues, eigenvectors = [], []
    for i in tqdm(range(len(sys.H)), desc=f"{'finding eigenstates':<35}"):
        eigenvalue, eigenvector = eigenstates(sys.H[i])  # Compute once
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)  
    return [np.array(eigenvalues), np.array(eigenvectors)] 

def select_states(sys, selection='ground'):
    """ Custructs sets of states from a set of eigenstates.
    
        Parameters
        ----------
        vars            : 1D array; typically a range of coupling strengths
        states          : 3D array; standard representation
        quantum_numbers : 3D array; standard representation
        selection       : list or string
            list        : [[<index for each state by sorted eigenvalue>], [<the same but for the other parity>]]
            strings     : "ground" yields the ground state for each parity
                          "random" yields a single state as a weighted superposition of eigenstates
     
        Returns
        -------
        states          : 3D array; standard representation
        quantum_numbers : 3D array; standard representation """
    
    # Manual state selection
    if type(selection) == list:
    
        # Consolidate lists
        if type(selection[0]) == list:
            selection_cache = []
            for i in range(len(selection[0])):
                selection_cache.append(selection[0][i])
            for i in range(len(selection[1])):
                selection_cache.append(int(len(sys.states[0][0])/2)+selection[1][i])
            selection = selection_cache
        
        # Update and return states
        states          = [sys.states[0][:, selection], sys.states[1][:, :, selection]]
        quantum_numbers = sys.quantum_numbers[:, selection]
        return states, quantum_numbers
    
    # Ground states
    elif selection == 'ground':
    
        # Choose ground state for each parity
        selected_states = [0, int(len(sys.states[0][0])/2)]
        
        # Update and return states
        states          = [sys.states[0][:,selected_states], sys.states[1][:,:,selected_states]]
        quantum_numbers = sys.quantum_numbers[:,selected_states]
        return states, quantum_numbers
    
    # Random state
    elif selection == 'random':
    
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
        for i in range(len(sys.states[1])):
        
            # Construct state
            new_state = np.zeros_like(sys.states[1][0][:,0])
            for j in range(num_eigenstates):
                new_state += random_weights[j] * sys.states[1][i][:,random_eigenstates[j]]
            
            # Normalize and recast as column vector
            new_state = (new_state / row_to_column(np.linalg.norm(new_state)))
            
            # Calculate energy
            new_energy = expectation(sys.H(sys.vars[i]), new_state)
            
            # Append to data container
            new_states[0].append([new_energy])
            new_states[1].append(new_state)
        
        new_states = [np.array(new_states[0]), np.array(new_states[1])]
        return new_states, None

def sort_by_quantum_numbers(sys, sort=None, secondary_sort=None, sort_dict={}):
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

    # Find numbers for each state
    expectation_list = calculate_quantum_numbers(sys, sort, secondary_sort, sort_dict)

    # Loop over each set of states
    for i in range(len(expectation_list)):
        row = expectation_list[i]
        
        # Sort by secondary eigenvalue parameter first (if provided)
        if secondary_sort:
            sorted_indices = np.argsort(row[:, 1], kind='stable')
        else:
            sorted_indices = np.arange(len(row))  # Default indices if no secondary sort
        
        # Then apply a stable sort by the primary eigenvalue parameter, preserving secondary order
        sorted_indices = sorted_indices[np.argsort(row[sorted_indices, 0], kind='stable')]

        sorted_row = row[sorted_indices]  # Sort expectation_list row
        sorted_states_0.append(np.array(sys.states[0][i])[sorted_indices])  # Sort eigenvalues in states[0][i]
        sorted_states_1.append(sys.states[1][i][:, sorted_indices])  # Sort states based on sorted indices
        sorted_expectations.append(sorted_row)  # Store sorted expectations

    sorted_states_0 = np.array(sorted_states_0)  # Convert list to array for consistency
    sorted_states_1 = np.array(sorted_states_1)  # Convert list to array for consistency

    return [sorted_states_0, sorted_states_1], np.array(sorted_expectations)

def calculate_quantum_numbers(sys, sort=None, secondary_sort=None, sort_dict={}, precision=10):
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
        sort_dict = {'P': sys.P, 'n': sys.a_dag @ sys.a, 'J_z': sys.J_z}
    
    # Set rounding
    set_precision = precision
    
    # Cycle through each λ
    expectation_list = []
    for i in tqdm(range(len(sys.states[1])), desc=f"{'calculating numbers':<35}"):
        expectations_rounded = []
        
        # Calculate all quantum numbers (|P, n, J_z, E⟩)
        if sort == None:
            P_expectations   = [expectation(sys.P,             row_to_column(sys.states[1][i][:,j])) for j in range(len(sys.states[1][0][0]))]
            n_expectations   = [expectation(sys.a_dag @ sys.a, row_to_column(sys.states[1][i][:,j])) for j in range(len(sys.states[1][0][0]))]
            J_z_expectations = [expectation(sys.J_z,           row_to_column(sys.states[1][i][:,j])) for j in range(len(sys.states[1][0][0]))]
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
                    for k in range(len(sys.states[0][i])):
                        expectations_rounded.append([round(sys.states[0][i][k], set_precision)])
                    expectation_list.append(np.array(expectations_rounded))
                
                # Calculate quantum number
                else:
                    expectations_cache = [expectation(sort_dict[sort], row_to_column(sys.states[1][i][:,j])) for j in range(len(sys.states[1][0][0]))]
                    for k in range(len(sys.states[0][i])):
                        expectations_rounded.append([round(expectations_cache[k], set_precision)])
                    expectation_list.append(np.array(expectations_rounded))
            
            # Calculate two quantum numbers (|sort, secondary_sort⟩)
            else:
                
                # Calculate first number
                if sort == 'E':
                    cache_1 = sys.states[0][i]
                elif sort == 'P':
                    cache_1 = [round(expectation(sort_dict[sort], row_to_column(sys.states[1][i][:,j]))) for j in range(len(sys.states[1][0][0]))]
                else:
                    cache_1 = [expectation(sort_dict[sort], row_to_column(sys.states[1][i][:,j])) for j in range(len(sys.states[1][0][0]))]
                
                # Calculate second number
                if secondary_sort == 'E':
                    cache_2 = sys.states[0][i]
                else:
                    cache_2 = [expectation(sort_dict[secondary_sort], row_to_column(sys.states[1][i][:,j])) for j in range(len(sys.states[1][0][0]))]
                for k in range(len(sys.states[0][i])):
                    expectations_rounded.append([round(np.real(cache_1[k]), set_precision),
                                                 round(np.real(cache_2[k]), set_precision)])
                expectation_list.append(np.array(expectations_rounded))
    
    return np.array(expectation_list)

def plot_handling(results, quantum_numbers=None, plot_mode='2D', no_show=False):
    """ Initializes matplotlib and generates plots from input data.
        
        Parameters
        ----------
        results         : list of lists; [[titles_1, values_1, indices_1, style_1],
                                          [titles_2, values_2, indices_2, style_2], ...]
        quantum_numbers : 3D array; corresponding quantum numbers for coloring plots
        plot_mode       : string in {'2D', '3D'}; self-explanatory
        no_show         : Boolean; omits plt.show() if True
      
        Details
        -------
        titles  : (row_title, column_title); ex. ('x', 'f(x)')
        values  : (x_values,  y_values);     ex. (x_array, y_array)
        indices : (row_index, column_index); ex. (0, 2)
        style   : (plot_type);               ex. ('plot') or ('contour') """
    
    # 2D plotting
    if plot_mode == '2D':
    
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
        if not no_show: plt.show()
    
    # 3D plotting
    elif plot_mode == '3D':
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
        if not no_show: plt.show()

def tolerance_check(array, tolerance=1e-8):
    """ Sends each value in an array to zero if it is less than a set tolerance. """

    # Test real and imaginary parts
    real_part = np.where(np.abs(array.real) <= tolerance, 0, array.real)
    imag_part = np.where(np.abs(array.imag) <= tolerance, 0, array.imag)

    # Convert to a real-valued array if the imaginary part is null
    if np.all(imag_part == 0): return real_part
    else:                      return real_part + 1j * imag_part

########################################################################################################################################################
# Classes
class System:
    """ Creates and stores data, including the Hamiltonian and its eigenstates. """

    @classmethod
    def load(cls, filename='cache'):
        """ Load states with sys=System.load(filename).
            If a system is already initialized, use sys=sys.load(filename). 
            
            Parameters
            ----------
            filename : string, such as 'data' """
    
        with open('Dicke_' + filename + '.pkl', 'rb') as file:
            system_instance = pickle.load(file)
        return system_instance

    def __init__(self, ω=1, ω0=1, n_max=2, N=2, ℏ=1, m=1, spin=1/2, debug=False, indiv=False):
        """ Initializes operators and parameters.
        
            Parameters
            ----------
            ω     : float; field frequency
            ω0    : float; atomic frequency
            n_max : integer; number of field modes
            N     : integer; number of particles
            
            ℏ     : float; Planck's constant
            m     : float; mass
            spin  : integer or half-integer; atomic spin
            
            debug : Boolean; sets a default variable and finds states
            indiv : Boolean; constructs J with individual Pauli matrices, rather than collective """
    
        # Initialize parameters
        self.ω                      = ω
        self.ω0                     = ω0
        self.n_max                  = n_max
        self.N                      = N
        
        self.ℏ                      = ℏ
        self.m                      = m
        self.spin                   = spin
        
        self.crit                   = (ω * ω0)**(1/2)
        self.J                      = N/2
        self.indiv                  = indiv
        if indiv: self.m_J          = N**2
        else:     self.m_J          = int(2*self.J + 1)

        # Create independent spaces
        J_spin_dict                 = create_J_operators(self, individual=self.indiv)
        a_field_dict                = create_a_operators(self)
        self.J_x_spin               = J_spin_dict['J_x']
        self.J_y_spin               = J_spin_dict['J_y']
        self.J_z_spin               = J_spin_dict['J_z']
        self.J_m_spin               = J_spin_dict['J_m']
        self.J_p_spin               = J_spin_dict['J_p']
        self.a_field                = a_field_dict['a']
        self.a_dag_field            = a_field_dict['a_dag']
        del J_spin_dict, a_field_dict
        
        # Create tensor product spaces
        J_a_dict = compute_tensor_products(self)
        self.J_x                    = J_a_dict['J_x']              # spin x operator
        self.J_y                    = J_a_dict['J_y']              # spin y operator
        self.J_z                    = J_a_dict['J_z']              # spin z operator
        self.J_m                    = J_a_dict['J_m']              # spin annihilation operator
        self.J_p                    = J_a_dict['J_p']              # spin creation operator
        self.a                      = J_a_dict['a']                # field annihilation operator
        self.a_dag                  = J_a_dict['a_dag']            # field creation operator
        self.P                      = create_parity_operator(self) # parity operator
        del J_a_dict

        # Initialize things to be assigned later
        self.vars                   = None                         # variable array
        self.states                 = None                         # selected eigenstates
        self.states_backup          = None                         # complete set of eigenstates
        self.quantum_numbers        = None                         # selected eigenvalues
        self.quantum_numbers_backup = None                         # complete set of eigenvalues
        self.H                      = None                         # Hamiltonian
        self.desc                   = None                         # short description for saved files
        
        # Cache/other
        self.plot_list = None                                      # last plotted results

        # Set variable and find states
        if debug:
            self.Hamiltonian(np.linspace(0, self.crit, 1))
            self.sort()
        else:
            self.variable()

    def variable(self):
        """ Creates a variable and the associated Hamiltonians.
            See Hamiltonian() for more details. """
        
        lower     = float(input(f"{'variable lower bound (N * λ_crit)':<35}: "))
        upper     = float(input(f"{'variable upper bound (N * λ_crit)':<35}: "))
        samples   = int(input(f"{'number of trials':<35}: "))
        
        modifiers = [item.strip() for item in input(f"{'modifiers (csv)':<35}: ").split(',')]
        
        self.Hamiltonian(np.linspace(lower*self.crit, upper*self.crit, samples), modifiers)
        self.sort()

    def Hamiltonian(self, vars, modifiers='Dicke'):
        """ Creates a Hamiltonian for each variable in the set, then finds eigenstates.
        
            Parameters
            ----------
            vars : 1D array; see calculate_states() for more details """

        # Standard Dicke
        if 'Dicke' or 'None' in modifiers:
            H_field = self.ℏ * self.ω  * (self.a_dag @ self.a)
            H_atom  = self.ℏ * self.ω0 * self.J_z
            H_int   = self.ℏ / np.sqrt(self.N) * (self.a + self.a_dag) @ self.J_x
            H       = lambda λ: H_field + H_atom + λ*H_int
        
        # Dicke + Ising
        elif 'Ising' in modifiers:
            H_field = self.ℏ * self.ω  * (self.a_dag @ self.a)
            H_atom  = self.ℏ * self.ω0 * self.J_z
            H_Ising = self.ℏ * self.ω0 * self.J_z @ self.J_z
            H_int   = self.ℏ / np.sqrt(self.N) * (self.a + self.a_dag) @ self.J_x
            H       = lambda λ: H_field + H_atom + H_Ising + λ*H_int
        
        # Calculate Hamiltonian
        H_list = []
        for i in range(len(vars)):
            H_list.append(H(vars[i]))
        del H_field, H_atom, H_int, H
        
        # Assign results to the system
        self.vars   = vars
        self.H      = np.array(H_list)
        self.states = calculate_states(self)

    def sort(self, sort_1=None, sort_2=None):
        """ Sorts states by eigenvalue.
        
            Parameters
            ----------
            sort_1 : string; see sort_by_quantum_numbers() for more details
            sort_2 : string; see sort_by_quantum_numbers() for more details """
    
        # Set default values
        if sort_1 is None and sort_2 is None:
            sort_1 = 'P'
            sort_2 = 'E'        
        
        self.states, self.quantum_numbers = sort_by_quantum_numbers(self, sort_1, sort_2)

    def select(self, selection=None):
        """ Manages the states used for plotting and analysis.

            Parameters
            ----------
            selection : string or list of integers; see select for details """
        
        # Name options
        options = ['backup', 'restore']
        
        # Create a backup
        if (selection == 'backup') or (self.states_backup == None):
            self.states_backup          = self.states
            self.quantum_numbers_backup = self.quantum_numbers
        
        # Restore a backup
        if (selection == 'restore') or (self.states[0].shape != self.states_backup[0].shape):
            self.states          = self.states_backup
            self.quantum_numbers = self.quantum_numbers_backup
        
        # Select states
        if (selection != None) and (selection not in options):
            self.states, self.quantum_numbers = select_states(self, selection)

    def restore(self):
        """ A shortcut for sys.select('restore') """
        self.select('restore')

    def plot(self, selection='spectrum'):
        """ Just a convenient shorthand for plotting; see plot_data for more details. """        
        plot_data(self, selection)

    def update(self):
        """ Run this every time a parameter is changed. """
        self.__init__(self.ω, self.ω0, self.n_max, self.N, self.ℏ, self.m, self.spin)
        self.Hamiltonian(self.vars)
        self.sort('P', 'E')
    
    def save(self, filename='cache'):
        """ Save states with sys.save('filename'). """
        filename = 'Dicke_' + filename + '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def print(self, data='parameters'):
        if data == 'parameters':
            print(f"\n------------------------------------------\n"
                  f"{'field frequency':<35}: {round(self.ω, 2):<4}|\n"
                  f"{'atomic frequency':<35}: {round(self.ω0, 2):<4}|\n"
                  f"{'critical coupling':<35}: {round(self.crit, 2):<4}|\n"
                  f"{'number of modes':<35}: {self.n_max:<4}|\n"
                  f"{'number of particles':<35}: {self.N:<4}|\n"
                  f"------------------------------------------\n")
        
        elif data == 'numbers':
            try:
                print(f"\n-----------------------------------------------------------")
                print(f"|P, E)\t\t |P, E)\t\t |P, E)\t\t |P, E)")
                for i in range(len(self.quantum_numbers[0])//4):
                    print(f"|{round(self.quantum_numbers[0][4*i][0])}, {round(self.quantum_numbers[0][4*i][1], 2)})"
                        f"\t |{round(self.quantum_numbers[0][4*i+1][0])}, {round(self.quantum_numbers[0][4*i+1][1], 2)})"
                        f"\t |{round(self.quantum_numbers[0][4*i+2][0])}, {round(self.quantum_numbers[0][4*i+2][1], 2)})"
                        f"\t |{round(self.quantum_numbers[0][4*i+3][0])}, {round(self.quantum_numbers[0][4*i+3][1], 2)}) |")
                if len(self.quantum_numbers[0])/4 - len(self.quantum_numbers[0])//4 != 0:
                    for i in range(len(self.quantum_numbers[0]) - len(self.quantum_numbers[0]//4)):
                        print(f"|{self.quantum_numbers[0][i][0]}, {round(self.quantum_numbers[0][i][1], 2)}⟩\t")
                print(f"-----------------------------------------------------------\n")
            except:
                pass
        
        elif data == 'all numbers':
            print(self.quantum_numbers)

########################################################################################################################################################
# WIP
def Chebyshift(sys):
    from scipy.special import jv  # Bessel function of the first kind
    from scipy.sparse import identity, csr_matrix
    from scipy.sparse.linalg import eigsh, LinearOperator
    
    def chebyshev_time_evolution(H, input_state, t, num_terms):
    
        # Estimate the max and min eigenvalues of H
        E_min, E_max = eigsh(H, k=2, which='BE', return_eigenvectors=False)
        
        psi_cache = np.zeros_like(input_state, dtype=np.complex128)
        input_state = psi_cache + input_state

        # Scale the Hamiltonian
        H_scaled = (2 * H - (E_max + E_min) * csr_matrix(identity(H.shape[0]))) / (E_max - E_min)

        # Initial Chebyshev polynomials
        T0 = input_state
        T1 = H_scaled @ input_state
        
        # Time evolution result (initialized with the first term)
        output_state = jv(0, t * (E_max - E_min) / 2) * T0
        
        # Iteratively compute higher-order terms
        for i in range(1, num_terms):
        
            Tn = np.zeros(T0.shape[0], dtype=np.complex128).reshape((input_state.shape[0], 1))
            Tn += 2 * (H_scaled @ T1) - T0

            # Add the contribution of the nth term
            output_state += (2 * (-1j)**i * jv(i, t * (E_max - E_min) / 2)) * Tn
            
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
    for i in tqdm(range(len(sys.states[1])), desc=f"{'calculating evolution':<35}"):
        expectation_values = []
        
        # Cycle through states
        for j in range(states[1][i].shape[1]):
            expectation_values.append([])

            # Extract column vector
            state = row_to_column(sys.states[1][i][:,j])
            
            # Evolve state
            for t in times:
            
                # Compute the time-evolved state at time t
                state_evolved = chebyshev_time_evolution(sys.H(sys.vars[i]), state, t, num_terms)
                state_evolved = state_evolved / np.linalg.norm(state_evolved)
                
                # Calculate a property of the evolved state (e.g., probability |state_evolved|^2)
                measure = expectation(sys.H(sys.vars[i]), state_evolved, single_state=True)
                
                # Store the total measure at this time step (or any other property of interest)
                expectation_values[j].append(measure)

        expectation_values = np.array(expectation_values).T
        plot_list.append([(f"$t,\tλ={round(sys.vars[i],2)}$", f"$⟨E⟩$"), 
                          (times, expectation_values), 
                          (0, i), 
                          ('plot')])
    return plot_list

def Lindbladian(sys):
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
    for i in tqdm(range(len(sys.vars)), desc=f"{'calculating Lindbladian':<35}"):

        # Initialize data container for plotting
        expectation_values = []

        # Generate density matrices
        ρ_array = []
        for j in range(sys.states[1][i].shape[1]):
            ρ_array.append(np.outer(sys.states[1][i][:,j], sys.states[1][i][:,j].conj()))
        ρ_array = np.array(ρ_array)

        # Sort through density matrices
        for j in range(len(ρ_array)):
            ρ = ρ_array[j]
            expectation_values.append([])

            # Sort through each time step
            for t in range(len(times)):

                # Store observable for plotting
                expectation_values[j].append(np.real(np.trace(sys.J_z @ ρ)))

                # Construct the Lindbladian and evolve the density matrix
                dρ = -1j * (sys.H(sys.vars[i]) @ ρ - ρ @ sys.H(sys.vars[i]))
                for M in L:
                    anticommutator = (M.conj().T @ M) @ ρ + ρ @ (M.conj().T @ M)
                    dρ += M @ (ρ @ M.conj().T) - (1/2) * anticommutator
                ρ = ρ + dt * dρ
                
        expectation_values = np.array(expectation_values).T
        plot_list.append([(f"$t, λ={round(sys.vars[i],2)}$", f"$⟨J_z⟩$"), 
                          (times, expectation_values), 
                          (0, i), 
                          ('plot')])
    return plot_list

def SEOP_Dicke_model():
    """ Spin-exchange optical pumping model
    
        Hamiltonian
        -----------
        ordered by strength : aI∙S + gμS∙B + μI∙B + μK∙B + γN∙S + aK∙S + bK∙(3R^2-1)∙S
        ordered by glamour  : (aI + γN + aK)∙S + (gμS + μI + μK)∙B + bK∙(3R^2-1)∙S """
        
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
    H_field = sys.ℏ * ω  * (a_dag @ a)
    H_I     = sys.ℏ * ω0 * I_z
    H_S     = sys.ℏ * ω0 * S_z
    H_spin  = sys.ℏ * ω0 * I_z @ S_z
    H_int   = 2 * sys.ℏ / np.sqrt(N) * (a + a_dag) @ S_x
    H       = lambda λ: H_field + H_I + H_S + H_spin + λ*H_int

    # Generate all eigenstates and eigenvalues
    vars = np.linspace(0, 10*λ_critical, 101)
    states       = calculate_states(vars)

    # Sort eigenstates and eigenvalues
    sort_dict = {'P': P, 'E': H, 'S_z': S_z}
    states, quantum_numbers = sort_by_quantum_numbers(states, sort='P', secondary_sort='E')

    # Define custom plotting
    def SEOP_occupation(vars, states, quantum_numbers):
    
        # Select specific eigenstates
        selected_states = [0, int(len(states[0][0])/2)]
        states          = [states[0][:,selected_states], states[1][:,:,selected_states]]
        quantum_numbers = quantum_numbers[:,selected_states]
    
        n_expectations   = expectation(a_dag@a, states, single_state=False)
        J_x_expectations = expectation(I_z,     states, single_state=False)
        J_z_expectations = expectation(S_z,     states, single_state=False)
        
        plot_list = [[(f"$λ$", f"$⟨n⟩$"),   (vars, n_expectations),   (0, 1), ('plot')],
                     [(f"$λ$", f"$⟨I_z⟩$"), (vars, J_x_expectations), (1, 0), ('plot')],
                     [(f"$λ$", f"$⟨J_z⟩$"), (vars, J_z_expectations), (1, 2), ('plot')]]
        return plot_list
    
    # Make a calculation
    spectrum_plot_list   = find_spectrum(vars, states, quantum_numbers)
    occupation_plot_list = SEOP_occupation(vars, states, quantum_numbers)
    plot_handling(spectrum_plot_list,   quantum_numbers)
    plot_handling(occupation_plot_list, quantum_numbers)

def two_spin_Dicke_model():
    """ Spin-exchange optical pumping model
    
        Hamiltonian
        -----------
        ordered by strength : aI∙S + gμS∙B + μI∙B + μK∙B + γN∙S + aK∙S + bK∙(3R^2-1)∙S
        ordered by glamour  : (aI + γN + aK)∙S + (gμS + μI + μK)∙B + bK∙(3R^2-1)∙S """
        
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
    H_field = sys.ℏ * ω  * (a_dag @ a)
    H_spin  = sys.ℏ * ω0 * (S_z + I_z)
    H_int   = 2 * sys.ℏ / np.sqrt(N) * (a + a_dag) @ (S_x - I_x)
    H       = lambda λ: H_field + H_spin + λ*H_int

    # Generate all eigenstates and eigenvalues
    vars = np.linspace(0, 5*λ_critical, 101)
    states       = calculate_states(vars)

    # Sort eigenstates and eigenvalues
    sort_dict = {'P': P, 'E': H, 'S_z': S_z, 'n': a_dag@a}
    states, quantum_numbers = sort_by_quantum_numbers(states, sort='P', secondary_sort='E')

    # Define custom plotting
    def plot_n_S(vars, states, quantum_numbers):
    
        # Select specific eigenstates
        selected_states = [0, int(len(states[0][0])/2)]
        states          = [states[0][:,selected_states], states[1][:,:,selected_states]]
        quantum_numbers = quantum_numbers[:,selected_states]
    
        n_expectations   = expectation(a_dag@a, states, single_state=False)
        J_x_expectations = expectation(I_z,     states, single_state=False)
        J_z_expectations = expectation(S_z,     states, single_state=False)
        
        plot_handling([[(f"$λ$", f"$⟨n⟩$"),   (vars, n_expectations),   (0, 1), ('plot')],
                      [(f"$λ$", f"$⟨I_z⟩$"), (vars, J_x_expectations), (1, 0), ('plot')],
                      [(f"$λ$", f"$⟨J_z⟩$"), (vars, J_z_expectations), (1, 2), ('plot')]],
                      quantum_numbers = quantum_numbers)
    
    # Make a calculation
    #find_spectrum(vars, states, quantum_numbers)
    plot_n_S(vars, states, quantum_numbers)

########################################################################################################################################################
# Main
def main():
    examples(0)    

if __name__ == '__main__':
    main()

########################################################################################################################################################