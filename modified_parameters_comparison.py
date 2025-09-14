# This file uses the functions defined in the previous files
# It modifies the delta_R parameter and compares deterministic vs stochastic results

# Modified parameters (delta_R = 0.05)
modified_params = [
    50,   # alpha_A
    500,  # alpha_A_prime
    0.01, # alpha_R
    50,   # alpha_R_prime
    50,   # beta_A
    5,    # beta_R
    10,   # delta_MA
    0.5,  # delta_MR
    1,    # delta_A
    0.05, # delta_R (modified)
    1,    # gamma_A
    1,    # gamma_R
    2,    # gamma_C
    50,   # theta_A
    100   # theta_R
]

# Solve deterministic model with modified parameters
sol_deterministic_modified = solve_ivp(Genetic_oscillator, t_span, initial_conditions, 
                                      method='BDF', args=(modified_params,), t_eval=t_eval)

# Create modified parameters dictionary for stochastic simulation
modified_params_dict = params.copy()
modified_params_dict['delta_R'] = 0.05

# Run stochastic simulation with modified parameters
times_modified, states_modified = SSA_solver(initial_state, StateChangeMat, modified_params_dict, final_time)
R_values_stochastic_modified = states_modified[:, 7]

# Plot comparison
plt.figure(figsize=(12, 6))
plt.plot(sol_deterministic_modified.t, sol_deterministic_modified.y[7], 
         label='Deterministic R', color='blue', linewidth=2)
plt.plot(times_modified, R_values_stochastic_modified, 
         label='Stochastic R', color='orange', alpha=0.7)
plt.xlabel('Time (hours)')
plt.ylabel('Concentration of R')
plt.title('Comparison: Deterministic vs Stochastic (δ_R = 0.05)')
plt.legend()
plt.grid(True)
plt.savefig('comparison_deltaR_modified.png')
plt.show()
