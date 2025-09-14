import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def Genetic_oscillator(t, y, params):
    D_A, D_R, D_A_prime, D_R_prime, M_A, A, M_R, R, C = y
    alpha_A, alpha_A_prime, alpha_R, alpha_R_prime, beta_A, beta_R, delta_MA, delta_MR, delta_A, delta_R, gamma_A, gamma_R, gamma_C, theta_A, theta_R = params

    dD_A_dt = theta_A * D_A_prime - gamma_A * D_A * A
    dD_R_dt = theta_R * D_R_prime - gamma_R * D_R * A
    dD_A_prime_dt = gamma_A * D_A * A - theta_A * D_A_prime
    dD_R_prime_dt = gamma_R * D_R * A - theta_R * D_R_prime
    dM_A_dt = alpha_A_prime * D_A_prime + alpha_A * D_A - delta_MA * M_A
    dA_dt = beta_A * M_A + theta_A * D_A_prime + theta_R * D_R_prime - A * (gamma_A * D_A + gamma_R * D_R + gamma_C * R + delta_A)
    dM_R_dt = alpha_R_prime * D_R_prime + alpha_R * D_R - delta_MR * M_R
    dR_dt = beta_R * M_R - gamma_C * A * R + delta_A * C - delta_R * R
    dC_dt = gamma_C * A * R - delta_A * C

    return [dD_A_dt, dD_R_dt, dD_A_prime_dt, dD_R_prime_dt, dM_A_dt, dA_dt, dM_R_dt, dR_dt, dC_dt]

# Parameters from the article
initial_parameters = [
    50,   # alpha_A: 1/h
    500,  # alpha_A_prime: 1/h
    0.01, # alpha_R: 1/h
    50,   # alpha_R_prime: 1/h
    50,   # beta_A: 1/h
    5,    # beta_R: 1/h
    10,   # delta_MA: 1/h
    0.5,  # delta_MR: 1/h
    1,    # delta_A: 1/h
    0.2,  # delta_R: 1/h
    1,    # gamma_A: 1/(mol*h)
    1,    # gamma_R: 1/(mol*h)
    2,    # gamma_C: 1/(mol*h)
    50,   # theta_A: 1/h
    100   # theta_R: 1/h
]

initial_conditions = [1, 1, 0, 0, 0, 0, 0, 0, 0]
t_span = (0, 400)
t_eval = np.linspace(0, 400, 1000)

# Solve with different methods
sol_RK45 = solve_ivp(Genetic_oscillator, t_span, initial_conditions, method='RK45', args=(initial_parameters,), t_eval=t_eval)
sol_BDF = solve_ivp(Genetic_oscillator, t_span, initial_conditions, method='BDF', args=(initial_parameters,), t_eval=t_eval)
sol_Radau = solve_ivp(Genetic_oscillator, t_span, initial_conditions, method='Radau', args=(initial_parameters,), t_eval=t_eval)

# Plot results
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.plot(sol_RK45.t, sol_RK45.y[5], label='Activator A')
plt.plot(sol_RK45.t, sol_RK45.y[7], label='Repressor R')
plt.title('RK45 Solver')
plt.xlabel('Time (hours)')
plt.ylabel('Concentration')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(sol_BDF.t, sol_BDF.y[5], label='Activator A')
plt.plot(sol_BDF.t, sol_BDF.y[7], label='Repressor R')
plt.title('BDF Solver')
plt.xlabel('Time (hours)')
plt.ylabel('Concentration')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(sol_Radau.t, sol_Radau.y[5], label='Activator A')
plt.plot(sol_Radau.t, sol_Radau.y[7], label='Repressor R')
plt.title('Radau Solver')
plt.xlabel('Time (hours)')
plt.ylabel('Concentration')
plt.legend()

plt.tight_layout()
plt.savefig('solver_comparison.png')
plt.show()
