# Dynamics Component

## Purpose
Aerodynamic force models applied to vehicles during simulation. Currently implements linear drag.

## Inputs
- Vehicle `State`: velocity (body frame)
- Simulation timestep `dt`

## Outputs
- Drag force vector `(3,)` (body frame)

## Dependencies
None (standalone, plugged into Vehicle via config).

## Key Files
- `drag.py` — Base `Drag` abstract class
- `linear_drag.py` — Linear drag model: `F_drag = -diag(coeffs) * v_body`

## Calling Contract
- `Drag.update(state, dt)`: Called by Vehicle every step. Returns force vector.
