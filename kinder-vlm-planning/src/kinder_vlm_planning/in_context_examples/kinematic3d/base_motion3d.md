**Example 1: Direct Path to Target**

Initial State:
- Robot: base at (0.0, 0.0, 0.0), orientation 0°
- Target: position (1.0, 0.5, 0.2), radius 0.05m

Goal: Move robot base to within 0.05m of target position (1.0, 0.5)

Strategy: Navigate directly to target using collision-free motion planning

High-Level Plan:
1. move_base_to_target(robot, target, params=[])
   - Navigate robot base from (0.0, 0.0, 0.0) to (1.0, 0.5, 0.2)
   - No parameters needed (params=[])
   - Uses PyBullet motion planning internally to compute collision-free path
   - Plans sequence of base poses (x, y, theta) as waypoints
   - Controller executes waypoint trajectory:
     - Computes velocities [dx_base, dy_base, dtheta_base] for base motion
     - Maintains arm joints [7 values] and gripper [1 value] at fixed configuration
     - Action space: [dx_base, dy_base, dtheta_base, joint1, ..., joint7, gripper] (11D)
   - Simple case: path to (1.0, 0.5) with no obstacles

Goal Reached:
- Distance between robot base and target < 0.05m ✓
- Efficient navigation with automatic path planning ✓

Key Insights:
- move_base_to_target handles full motion planning pipeline internally
- No parameters required (empty tuple params=[])
- Controllers track planned waypoints via low-level actions
- Base controller maintains arm/gripper configuration during navigation

**Example 2: Target Behind Robot**

Initial State:
- Robot: base at (1.0, 1.0, 0.0), orientation 0° (facing right)
- Target: position (-0.5, 0.5, 0.2), radius 0.05m

Goal: Move robot base to within 0.05m of target position (-0.5, 0.5)

Strategy: Use motion planner to compute path (may include rotation/reversal)

High-Level Plan:
1. move_base_to_target(robot, target, params=[])
   - Navigate from (1.0, 1.0, 0.0) to (-0.5, 0.5, 0.2)
   - Motion planner computes efficient path:
     - Option A: Move backwards along trajectory
     - Option B: Rotate then drive forward
     - Option C: Combined motion optimizing smoothness
   - Controller executes planned waypoints
   - Base actions [dx_base, dy_base, dtheta_base] implement trajectory
   - Distance: √((1.0-(-0.5))² + (1.0-0.5)²) ≈ 1.58m
