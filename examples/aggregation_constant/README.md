# Example: Homogeneous Batch Aggregation

Validates the `pbe_cuda` aggregation kernel against the analytical
Smoluchowski solution for the constant kernel.

## Physical problem

A spatially homogeneous, well-mixed suspension undergoes pure aggregation
with a constant collision frequency function β(u, v) = β₀.

Starting from a monodisperse initial condition (all particles in the
smallest bin), the total number concentration N_tot(t) decays as:

N_tot(t) = N₀ / (1 + t / t_half),    t_half = 2 / (β₀ N₀)

This is the exact Smoluchowski solution and is used to validate the
numerical result.

## Setup

| Parameter | Value |
|---|---|
| Bins | 256 (geometric grid, ratio r = 2^(1/3)) |
| v_min | 1×10⁻¹⁸ m³ |
| β₀ | 1×10⁻¹⁷ m³/s |
| N₀ | 1×10¹⁴ #/m³ |
| Integrator | Explicit Euler |
| Steps | 2000 |

## Build and run

```bash
# From the repo root
cmake --build build --target example_homogeneous_batch
./build/examples/homogeneous_batch/example_homogeneous_batch
```

## Note on the Euler integrator

The example uses explicit Euler for clarity. In a real CFD or reactor
simulation, the library RHS function would typically be called from a
higher-order integrator (RK4, CVODE, etc.) implemented by the user.