# Example: Homogeneous Batch Breakage

Validates the `pbe_cuda` breakage kernel against two independent
analytical quantities for linear selection + symmetric binary daughter.

## Physical problem

A spatially homogeneous suspension undergoes pure breakage with:

- **Selection function**: S(v) = S₀ v / v_ref  (linear)
- **Daughter distribution**: symmetric binary — each breakage event
  produces exactly two fragments of size v/2

## Analytical solution

For a monodisperse initial condition N(v,0) = N₀ δ(v − v₀):

| Quantity | Analytical expression | Physical meaning |
|---|---|---|
| N_tot(t) | N₀ · exp(S₀ · t) | Exponential growth |
| V_tot(t) | N₀ · v₀ = const | Volume conservation |

## Setup

| Parameter | Value |
|---|---|
| Bins | 128 (geometric, r = 2^(1/3)) |
| IC bin | 100 (near top of grid) |
| S₀ | 1×10⁻³ s⁻¹ |
| v_ref | v₀ (pivot volume of IC bin) |
| N₀ | 1×10¹⁴ #/m³ |
| Quadrature | 1 point: t_q = 0.5, bw_q = 2.0 |
| Integrator | Explicit Euler, dt = 0.1 s |

## Build and run

```bash
cmake --build build --target example_homogeneous_batch_breakage
./build/examples/homogeneous_batch_breakage/example_homogeneous_batch_breakage
```

## Expected output

Two validation columns are printed at each output time:
- `Err N_tot` — relative error vs. analytical exponential growth
- `Err V_tot` — relative error in volume conservation (should be ~machine precision)