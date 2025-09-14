import numpy as np
import matplotlib.pyplot as plt

# Parameters dictionary
params = {
    'alpha_A': 50,
    'alpha_A_prime': 500,
    'alpha_R': 0.01,
    'alpha_R_prime': 50,
    'beta_A': 50,
    'beta_R': 5,
    'delta_MA': 10,
    'delta_MR': 0.5,
    'delta_A': 1,
    'delta_R': 0.2,
    'gamma_A': 1,
    'gamma_R': 1,
    'gamma_C': 2,
    'theta_A': 50,
    'theta_R': 100
}

initial_state = [1, 1, 0, 0, 0, 0, 0, 0, 0]

# State Change Matrix (16 reactions x 9 species)
StateChangeMat = np.array([
    [0, 0, 0, 0, 0, -1, 0, -1, +1],  # A + R -> C
    [0, 0, 0, 0, 0, -1, 0, 0, 0],     # A -> ∅
    [0, 0, 0, 0, 0, 0, 0, +1, -1],    # C -> R
    [0, 0, 0, 0, 0, 0, 0, -1, 0],     # R -> ∅
    [-1, 0, +1, 0, 0, -1, 0, 0, 0],   # D_A + A -> D'_A
    [0, -1, 0, +1, 0, -1, 0, 0, 0],   # D_R + A -> D'_R
    [+1, 0, -1, 0, 0, +1, 0, 0, 0],   # D'_A -> A + D_A
    [0, 0, 0, 0, +1, 0, 0, 0, 0],     # D_A -> D_A + M_A
    [0, 0, 0, 0, +1, 0, 0, 0, 0],     # D'_A -> D'_A + M_A
    [0, 0, 0, 0, -1, 0, 0, 0, 0],     # M_A -> ∅
    [0, 0, 0, 0, 0, +1, 0, 0, 0],     # M_A -> A + M_A
    [0, +1, 0, -1, 0, +1, 0, 0, 0],   # D'_R -> A + D_R
    [0, 0, 0, 0, 0, 0, +1, 0, 0],     # D_R -> D_R + M_R
    [0, 0, 0, 0, 0, 0, +1, 0, 0],     # D'_R -> D'_R + M_R
    [0, 0, 0, 0, 0, 0, -1, 0, 0],     # M_R -> ∅
    [0, 0, 0, 0, 0, 0, 0, +1, 0]      # M_R -> M_R + R
])

def PropensityFunc(state, params):
    D_A, D_R, D_A_prime, D_R_prime, M_A, A, M_R, R, C = state
    return [
        params['gamma_C'] * A * R,           # r1
        params['delta_A'] * A,                # r2
        params['delta_A'] * C,                # r3
        params['delta_R'] * R,                # r4
        params['gamma_A'] * D_A * A,          # r5
        params['gamma_R'] * D_R * A,          # r6
        params['theta_A'] * D_A_prime,        # r7
        params['alpha_A'] * D_A,              # r8
        params['alpha_A_prime'] * D_A_prime,  # r9
        params['delta_MA'] * M_A,             # r10
        params['beta_A'] * M_A,               # r11
        params['theta_R'] * D_R_prime,        # r12
        params['alpha_R'] * D_R,              # r13
        params['alpha_R_prime'] * D_R_prime,  # r14
        params['delta_MR'] * M_R,             # r15
        params['beta_R'] * M_R                # r16
    ]

def RandExp(a):
    return -np.log(np.random.random()) / a

def RandDist(react, probs):
    return np.random.choice(react, p=probs)

def SSA_solver(initial_state, StateChangeMat, params, final_time):
    m, n = StateChangeMat.shape
    ReactNum = np.arange(m)
    state = np.array(initial_state, dtype=float)
    times = [0]
    states = [state.copy()]
    t = 0

    while t < final_time:
        w = PropensityFunc(state, params)
        a = np.sum(w)
        tau = RandExp(a)
        t += tau
        if t > final_time:
            break
        probs = np.array(w) / a
        which = RandDist(ReactNum, probs)
        state += StateChangeMat[which, :]
        times.append(t)
        states.append(state.copy())

    return np.array(times), np.array(states)

# Run simulation
final_time = 400
times, states = SSA_solver(initial_state, StateChangeMat, params, final_time)

# Plot results
A_values = states[:, 5]
R_values = states[:, 7]

plt.figure(figsize=(12, 6))
plt.plot(times, A_values, label='Activator A', color='b')
plt.xlabel('Time (hours)')
plt.ylabel('Concentration')
plt.title('Concentration of A Over Time (Stochastic)')
plt.legend()
plt.grid(True)
plt.savefig('stochastic_A.png')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(times, R_values, label='Repressor R', color='r')
plt.xlabel('Time (hours)')
plt.ylabel('Concentration')
plt.title('Concentration of R Over Time (Stochastic)')
plt.legend()
plt.grid(True)
plt.savefig('stochastic_R.png')
plt.show()
