This project implements and compares deterministic and stochastic models of a genetic oscillator based on the activator-repressor mechanism described in Vilar et al. (2002). The models simulate circadian rhythm dynamics in biological systems, exploring how noise affects oscillatory behavior.

Deterministic Model (ODE)
The deterministic model uses a system of 9 ordinary differential equations to describe the dynamics between:

Activator protein (A) and its gene forms (Dₐ, D'ₐ)

Repressor protein (R) and its gene forms (Dᵣ, D'ᵣ)

mRNA molecules (Mₐ, Mᵣ)

Activator-Repressor complex (C)

Stochastic Model (SSA)
The stochastic model implements the Gillespie Algorithm to simulate the same system through 16 discrete biochemical reactions. This approach captures intrinsic noise and random fluctuations that are significant at cellular scales.
