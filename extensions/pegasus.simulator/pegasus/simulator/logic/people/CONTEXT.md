# People Component

## Purpose
Pedestrian simulation system for spawning and controlling human actors in the simulation environment. Supports waypoint-based movement controllers.

## Inputs
- USD person model paths
- Waypoint sequences for movement controllers
- Simulation timestep

## Outputs
- Animated person actors in the simulation stage
- Position updates per step

## Dependencies
None (standalone system, managed by PeopleManager).

## Key Files
- `person.py` — Person actor class (USD spawning and animation)
- `person_controller.py` — Base `PersonController` abstract class
- `line_person_controller.py` — Linear waypoint-following controller
- `../people_manager.py` — PeopleManager singleton (tracks all spawned people)

## Calling Contract
- `Person.update(dt)`: Called every step by PeopleManager. Advances controller.
- `PersonController.update(dt)`: Returns desired position/orientation for the person.
