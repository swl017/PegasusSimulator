# Thrusters Component

## Purpose
Motor thrust curve models that convert rotor angular velocities to forces and torques.

## Inputs
- Rotor angular velocities (from backend motor commands)
- Vehicle configuration (number of rotors, positions, directions)

## Outputs
- Per-rotor thrust forces
- Per-rotor torques (reaction torque from spinning)

## Dependencies
None (standalone, plugged into Vehicle via config).

## Key Files
- `thrust_curve.py` — Base `ThrustCurve` abstract class
- `quadratic_thrust_curve.py` — Quadratic model: `F = k * omega^2`

## Calling Contract
- `ThrustCurve.thrust(input_reference)`: Called by Vehicle every step. Returns forces per rotor.
