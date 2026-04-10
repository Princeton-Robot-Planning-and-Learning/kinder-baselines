**Example 1: Single Object Pick and Place**

Initial State:
- Robot: base position (0.0, 0.0), orientation 0° (facing forward), arm retracted, gripper open
- Cube1: position (0.6, 0.0, 0.05) on ground, red color, size 0.02m
- Cupboard_1: position (1.5, 0.0, 0.0), shelf fixture with 3 shelves

Goal: Place cube1 on shelf 2 of cupboard_1

Task Specification:
- Initial state: on(cube1, ground_1_object_init_region)
- Goal state: on(cube1, cupboard_1_cube_1_goal_region)

Strategy: Pick up cube from ground → Navigate to cupboard → Place cube on target shelf

High-Level Plan:
1. pick_ground(robot:mujoco_tidybot_robot, cube1:mujoco_movable_object)[0.5, 0.0]
   - Navigate robot base to approach cube1 from distance 0.5m
   - Approach angle 0.0 radians means approaching from directly in front
   - Lower arm to grasp height near ground level (~0.05m above ground)
   - Close gripper to grasp cube1
   - Retract arm to safe carrying configuration
   
   Detailed Steps:
   a) Base navigation: Move from (0.0, 0.0) to approximately (0.1, 0.0) 
      - Target distance parameter (0.5) positions robot 0.5m from cube
      - Target rotation parameter (0.0) aligns robot to face cube head-on
   b) Arm motion planning: Plan trajectory from retracted config to pre-grasp pose
      - End effector target: (0.39m forward, 0.0m lateral, -0.35m down from base)
      - Orientation: gripper faces down (quaternion: 0.707, 0.707, 0, 0)
   c) Gripper closing: Activate gripper closure (gripper_pos → 0.0)
   d) Arm retraction: Return to home configuration to lift cube safely

2. place_ground(robot:mujoco_tidybot_robot, cube1:mujoco_movable_object, cupboard_1:mujoco_fixture)[0.9, 0.0, -1.571]
   - Navigate to cupboard while holding cube1
   - Position relative to cupboard for shelf placement
   - Extend arm to place cube on target shelf
   - Open gripper to release cube
   - Retract arm to home position
   
   Detailed Steps:
   a) Base navigation: Move from pick location to cupboard approach position
      - Target distance parameter (0.9) positions robot 0.9m from cupboard center
      - Y-offset parameter (0.0) centers robot laterally with cupboard
      - Rotation parameter (-1.571 ≈ -π/2) orients robot perpendicular to cupboard
      - Robot moves while carrying cube1 (collision checking disabled for held object)
   b) Arm motion planning: Plan trajectory to shelf placement position
      - End effector target: (0.7m forward, 0.0m lateral, 0.0m height)
      - Orientation: gripper horizontal (quaternion: 0.5, 0.5, 0.5, 0.5)
      - Position aligns with target shelf height and depth
   c) Gripper opening: Release gripper (gripper_pos → 1.0)
   d) Arm retraction: Return to home configuration

Goal Reached:
- Cube1 successfully placed on cupboard_1 shelf 2 ✓
- Object within goal region tolerance (±0.3m x, ±0.12m y, ±0.03m z) ✓

Key Insights:
- pick_ground parameters [distance, rotation] control robot approach to ground objects
  - Distance typically 0.5-0.7m provides optimal reach without collision
  - Rotation in range [-π, π] determines approach angle relative to object
- place_ground parameters [distance, y_offset, rotation] control placement relative to fixture
  - Distance ~0.9m for cupboard positioning balances reach and stability
  - Y-offset allows lateral placement adjustment along fixture
  - Rotation -π/2 is standard for approaching cupboard from side
- Robot must navigate collision-free while carrying objects
- Arm trajectories planned in task space, IK solved in PyBullet simulation
- Gripper states: 0.0 = fully closed, 1.0 = fully open

**Example 2: Multiple Objects Sequential Placement**

Initial State:
- Robot: base position (0.0, 0.0), orientation 0°, arm retracted, gripper open
- Cube1: position (0.6, -0.1, 0.05) on ground
- Cube2: position (0.65, 0.1, 0.05) on ground
- Cupboard_1: position (1.5, 0.0, 0.0)

Goal: Place both cubes on cupboard shelves

Task Specification:
- Initial state: on(cube1, ground_region_1), on(cube2, ground_region_2)
- Goal state: on(cube1, cupboard_shelf_left), on(cube2, cupboard_shelf_right)

Strategy: Pick and place each cube sequentially

High-Level Plan:
1. pick_ground(robot:mujoco_tidybot_robot, cube1:mujoco_movable_object)[0.6, 0.0]
   - Approach cube1 from 0.6m distance at 0° angle
   - Lower arm and grasp cube1
   - Lift to carrying configuration

2. place_ground(robot:mujoco_tidybot_robot, cube1:mujoco_movable_object, cupboard_1:mujoco_fixture)[0.85, -0.1, -1.571]
   - Navigate to cupboard with cube1
   - Y-offset -0.1m places cube on left side of shelf
   - Place and release cube1

3. pick_ground(robot:mujoco_tidybot_robot, cube2:mujoco_movable_object)[0.6, 0.0]
   - Return to ground area and approach cube2
   - Approach from 0.6m distance at 0° angle
   - Grasp and lift cube2

4. place_ground(robot:mujoco_tidybot_robot, cube2:mujoco_movable_object, cupboard_1:mujoco_fixture)[0.92, 0.1, -1.571]
   - Navigate to cupboard with cube2
   - Y-offset +0.1m places cube on right side of shelf
   - Place and release cube2

Goal Reached:
- Both cubes successfully placed on cupboard shelves ✓
- Objects spatially separated to avoid collision ✓

Key Insights:
- Sequential pick-and-place for multiple objects
- Y-offset parameter crucial for placing multiple objects without collision
- Distance parameter may vary (0.85m vs 0.92m) to optimize base positioning
- Base navigation automatically avoids objects on ground during returns

**Example 3: Alternative Approach Angles**

Initial State:
- Robot: base position (-0.1, -0.1), orientation 45°
- Cube1: position (0.5, 0.2, 0.05) on ground (reachable from multiple angles)
- Cupboard_1: position (1.5, 0.0, 0.0)

Goal: Place cube1 on cupboard

Strategy: Use non-zero approach angle for pick

High-Level Plan:
1. pick_ground(robot:mujoco_tidybot_robot, cube1:mujoco_movable_object)[0.5, 1.571]
   - Approach cube1 from distance 0.5m
   - Rotation parameter 1.571 (≈ π/2) approaches from side (90° angle)
   - This avoids obstacles that might be in direct approach path
   - Grasp mechanics identical regardless of approach direction

2. place_ground(robot:mujoco_tidybot_robot, cube1:mujoco_movable_object, cupboard_1:mujoco_fixture)[0.9, 0.0, -1.571]
   - Standard cupboard placement approach
   - Distance 0.9m, centered (y_offset=0.0), perpendicular orientation

Goal Reached:
- Cube1 placed successfully using alternative approach ✓
- Demonstrates navigation flexibility ✓

Key Insights:
- pick_ground rotation parameter enables approach from any angle
- Useful for obstacle avoidance or constrained spaces
- π/2 = side approach, π = rear approach, 0 = front approach
- Robot automatically computes collision-free base motion plans
- End effector grasp pose remains consistent regardless of base approach angle

**Strategy Selection Guidelines:**
- For objects on ground: Use pick_ground with distance ≈ 0.5m for optimal reach
- For cupboard placement: Use distance ≈ 0.9m and rotation ≈ -π/2
- For multiple objects: Adjust y_offset to prevent placement collisions (±0.1m separation)
- For obstacle avoidance: Vary rotation parameter in pick_ground to approach from clear path
- Navigation automatically handles collision avoidance with world boundaries and static fixtures
