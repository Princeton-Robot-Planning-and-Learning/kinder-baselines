**Example 1: Open Drawer, Pick Wiper, Sweep Objects**

Initial State:
- Robot: base position (1.45, -0.25), orientation 180° (facing kitchen island), arm retracted, gripper open
- Kitchen_island_drawer: closed, located at shelf 1 of kitchen island at (0.5, 0.0, 0.5)
- Wiper_0: position (0.35, 0.0, 1.0) on kitchen_island shelf 2, T-shaped tool with handle height 0.20m, head length 0.20m
- Cube_0: position (0.15, -0.05, 1.0) on kitchen_island shelf 2, size 0.01m, red color
- Cube_1: position (0.25, 0.03, 1.0) on kitchen_island shelf 2, size 0.01m, red color
- Cube_2: position (0.30, -0.02, 1.0) on kitchen_island shelf 2, size 0.01m, red color
- Cube_3: position (0.20, 0.08, 1.0) on kitchen_island shelf 2, size 0.01m, red color
- Cube_4: position (0.40, -0.06, 1.0) on kitchen_island shelf 2, size 0.01m, red color

Goal: Sweep all five cubes from shelf 2 into the opened drawer at shelf 1

Task Specification:
- Initial state:
  - in(wiper_0, wiper_init_region)
  - in(cube_0, blocks_init_region), in(cube_1, blocks_init_region), in(cube_2, blocks_init_region)
  - in(cube_3, blocks_init_region), in(cube_4, blocks_init_region)
  - on(robot, robot_kitchen_init_region)
- Goal state:
  - on(cube_0, kitchen_island_shelf_1_partition_1_region)
  - on(cube_1, kitchen_island_shelf_1_partition_1_region)
  - on(cube_2, kitchen_island_shelf_1_partition_1_region)
  - on(cube_3, kitchen_island_shelf_1_partition_1_region)
  - on(cube_4, kitchen_island_shelf_1_partition_1_region)

Strategy: Open drawer → Pick up wiper tool → Sweep all cubes into opened drawer

High-Level Plan:
1. open_drawer(robot:mujoco_tidybot_robot, wiper_0:mujoco_movable_object, kitchen_island_drawer_s1c1:mujoco_drawer, cube_0:mujoco_movable_object, cube_1:mujoco_movable_object, cube_2:mujoco_movable_object, cube_3:mujoco_movable_object, cube_4:mujoco_movable_object)[0.7, -3.142]
   - Navigate robot to kitchen island drawer
   - Grasp the drawer handle
   - Pull drawer open to allow object placement
   - Release handle and retract arm
   
   Detailed Steps:
   a) Base navigation: Move from (1.45, -0.25) to drawer approach position
      - Target distance parameter (0.7) positions robot 0.7m from drawer center
      - Target rotation parameter (-3.142 ≈ -π) orients robot to face drawer directly
      - Navigation avoids collision with kitchen island and other fixtures
   b) Arm motion planning: Plan trajectory to drawer handle grasp pose
      - End effector target: (0.07m forward, 0.3m lateral, -0.12m down from handle)
      - Orientation: gripper angled for handle (roll: -π - π/16, pitch: 0, yaw: -π/2)
      - Transform: DRAWER_TRANSFORM_TO_OBJECT for initial grasp
   c) Gripper closing: Activate gripper closure (gripper_pos → 0.0) to grasp handle
   d) Drawer opening motion: Pull drawer outward along slide rail
      - End effector moves from initial grasp (0.07m) to extended position (0.18m)
      - Transform: DRAWER_TRANSFORM_TO_OBJECT_END for final position
      - Smooth trajectory following to maintain handle grasp
      - Drawer opens approximately 0.11m to provide access
   e) Gripper opening: Release handle (gripper_pos → 1.0)
   f) Arm retraction: Return to home configuration safely
   
   Action Steps: 300 steps total
   - Navigation: ~100 steps to approach drawer
   - Grasp and open: ~150 steps for arm motion, grasping, pulling
   - Retraction: ~50 steps to return to home

2. pick_wiper(robot:mujoco_tidybot_robot, wiper_0:mujoco_movable_object, kitchen_island_drawer_s1c1:mujoco_drawer, cube_0:mujoco_movable_object, cube_1:mujoco_movable_object, cube_2:mujoco_movable_object, cube_3:mujoco_movable_object, cube_4:mujoco_movable_object)[0.7, -3.142]
   - Navigate to wiper location on kitchen_island shelf 2
   - Approach wiper with specific orientation for tool grasping
   - Grasp wiper handle
   - Lift wiper to carrying configuration
   
   Detailed Steps:
   a) Base navigation: Move from drawer location to wiper position
      - Target distance parameter (0.7) positions robot 0.7m from wiper
      - Target rotation parameter (-3.142 ≈ -π) aligns robot perpendicular to shelf
      - Base must position for reaching elevated shelf 2 (~1.0m height)
   b) Arm motion planning: Plan trajectory to wiper handle grasp
      - End effector target: Wiper handle at (0.35, 0.0, 1.0) world coordinates
      - Grasp transform: WIPER_TRANSFORM_TO_OBJECT offset (0.02m, 0, 0.03m)
      - Orientation: angled grip for T-shaped handle (roll: -π - π/16, pitch: 0, yaw: -π/2)
      - IK solution computed for 7-DOF arm to reach elevated position
   c) Gripper closing: Close gripper (gripper_pos → 0.0) around handle
   d) Arm retraction: Lift wiper and return to stable carrying configuration
      - Collision checking accounts for wiper geometry (handle + head)
      - Held object transform: base_link_to_held_obj computed for wiper
   
   Action Steps: 300 steps total
   - Navigation: ~100 steps to approach shelf 2
   - Grasp: ~120 steps for arm motion and grasping
   - Lift: ~80 steps to retract with tool

3. sweep(robot:mujoco_tidybot_robot, wiper_0:mujoco_movable_object, kitchen_island_drawer_s1c1:mujoco_drawer, cube_0:mujoco_movable_object, cube_1:mujoco_movable_object, cube_2:mujoco_movable_object, cube_3:mujoco_movable_object, cube_4:mujoco_movable_object)[0.55, -3.142]
   - Position robot for sweep motion across shelf 2
   - Execute sweeping motion with wiper to push all cubes
   - Cubes fall from shelf 2 into opened drawer at shelf 1 below
   
   Detailed Steps:
   a) Base navigation: Move to sweep starting position
      - Target distance parameter (0.55) positions robot 0.55m from sweep target
      - Target rotation parameter (-3.142 ≈ -π) maintains perpendicular orientation
      - Position allows full range of motion for sweep trajectory
   b) Sweeping motion execution: Multi-waypoint end-effector path
      - Waypoint 1: WIPER_SWEEP_TRANSFORM - (-0.05m, -0.1m, 0.025m) start position
        * Wiper positioned at edge of shelf 2, slightly left of cubes
        * End effector at ~1.0m height matching shelf surface
      - Waypoint 2: WIPER_SWEEP_TRANSFORM_END - (0.15m, 0.05m, 0.025m) mid-sweep
        * Wiper sweeps rightward and forward across shelf
        * Pushes cubes toward drawer opening
        * Smooth trajectory maintains contact with shelf surface
      - Waypoint 3: WIPER_SWEEP_TRANSFORM_END_2 - (0.28m, 0.15m, 0.025m) final position
        * Completes sweep motion
        * Ensures all cubes pushed off shelf into drawer below
   c) Follow end-effector path: Smooth trajectory execution
      - Use smoothly_follow_end_effector_path() for Cartesian motion
      - IK solved at each step to track waypoints
      - Held wiper collision geometry checked throughout motion
   d) Arm retraction: Return to home after sweep completion
   
   Action Steps: 200 steps total
   - Navigation: ~50 steps to sweep start position
   - Sweep motion: ~130 steps for multi-waypoint trajectory
   - Retraction: ~20 steps to stabilize

Goal Reached:
- Drawer successfully opened to receive objects ✓
- Wiper tool acquired for manipulation ✓
- All five cubes swept from shelf 2 into shelf 1 drawer ✓
- Objects within goal region tolerance inside drawer partition ✓

Key Insights:
- open_drawer parameters [distance, rotation] control approach to drawer fixture
  - Distance typically 0.6-0.8m for reaching drawer handle
  - Rotation ≈ -π positions robot facing drawer directly
  - Handle grasp uses specialized DRAWER_TRANSFORM_TO_OBJECT offset
  - Pulling motion extends along drawer slide rail direction
- pick_wiper parameters [distance, rotation] control approach to elevated shelf object
  - Distance 0.6-0.8m balances reach and stability for shelf 2 (~1.0m height)
  - Rotation ≈ -π maintains perpendicular approach to shelf
  - Wiper grasp uses WIPER_TRANSFORM_TO_OBJECT for T-handle geometry
  - Carrying requires collision checking for extended tool geometry
- sweep parameters [distance, rotation] control sweep execution positioning
  - Distance 0.5-0.7m provides optimal range for sweeping motion
  - Rotation ≈ -π maintains alignment with shelf edge
  - Multi-waypoint trajectory: start → mid-sweep → end position
  - Waypoints: WIPER_SWEEP_TRANSFORM → TRANSFORM_END → TRANSFORM_END_2
  - Smooth Cartesian path following ensures contact with shelf surface
- Object parameters include all entities for collision avoidance:
  - robot, wiper, drawer, and all cubes (cube_0 through cube_4)
- Action spaces: 11-dimensional continuous control
  - Base: [vx, vy, omega] for mobile navigation
  - Arm: 7-DOF joint velocities for manipulation
  - Gripper: 1-DOF for open/close
- Physics simulation: PyBullet used for motion planning and IK
  - Collision-free trajectory planning with RRT-based planner
  - Inverse kinematics for end-effector target poses
  - Dynamic simulation validates skill execution

**Example 2: Alternative Sweep Parameters**

Initial State:
- Similar configuration to Example 1
- Cubes distributed differently on shelf 2: more clustered on right side
- Drawer already open (pre-opened scenario)

Goal: Pick wiper and sweep cubes into drawer (drawer opening not needed)

Strategy: Skip drawer opening → Pick wiper → Sweep with adjusted parameters

High-Level Plan:
1. pick_wiper(robot:mujoco_tidybot_robot, wiper_0:mujoco_movable_object, kitchen_island_drawer_s1c1:mujoco_drawer, cube_0:mujoco_movable_object, cube_1:mujoco_movable_object, cube_2:mujoco_movable_object, cube_3:mujoco_movable_object, cube_4:mujoco_movable_object)[0.65, -3.142]
   - Approach wiper from 0.65m distance
   - Standard perpendicular orientation (-π rotation)
   - Grasp and lift wiper tool

2. sweep(robot:mujoco_tidybot_robot, wiper_0:mujoco_movable_object, kitchen_island_drawer_s1c1:mujoco_drawer, cube_0:mujoco_movable_object, cube_1:mujoco_movable_object, cube_2:mujoco_movable_object, cube_3:mujoco_movable_object, cube_4:mujoco_movable_object)[0.60, -3.0]
   - Approach from 0.60m distance (slightly closer)
   - Rotation parameter -3.0 (≈ -π + 0.14) provides slight angle adjustment
   - Angled approach better aligns with clustered cube distribution
   - Same multi-waypoint sweep trajectory
   - All cubes successfully pushed into drawer

Goal Reached:
- Wiper acquired ✓
- Alternative sweep parameters adapt to object distribution ✓
- All cubes in drawer ✓

Key Insights:
- Drawer opening can be skipped if drawer already accessible
- Sweep distance parameter adjustable: 0.5-0.7m range
- Sweep rotation parameter allows angle tuning: -π ± 0.2 radians
- Parameter adjustment adapts to different object clustering patterns
- Same waypoint trajectory achieves sweep regardless of approach angle

**Example 3: Detailed Parameter Space Understanding**

Parameter Ranges:
- open_drawer: [distance, rotation]
  - distance: typically 0.6-0.8m (optimal range for reaching drawer handle)
  - rotation: -π to π (standard -π for direct frontal approach)
- pick_wiper: [distance, rotation]
  - distance: typically 0.6-0.8m (balances reach for elevated shelf)
  - rotation: -π to π (standard -π for perpendicular shelf approach)
- sweep: [distance, rotation]
  - distance: typically 0.5-0.7m (provides sweep motion range)
  - rotation: -π to π (standard -π, slight variations for angle adjustment)

Action Space Details:
- TidyBot3D robot: 11-dimensional continuous action space
  - Base motion: [vx, vy, omega] ∈ [-0.1, 0.1]³ m/s and rad/s
  - Arm joints: 7-DOF velocities within joint limits (rad/s)
  - Gripper: 1-DOF position control ∈ [0.0, 1.0] (0=closed, 1=open)

Physical Constraints:
- Kitchen island fixture: fixed position, provides shelf 1 and shelf 2 surfaces
- Drawer: slide rail mechanism, opens ~0.11m outward
- Wiper: T-shaped tool, handle 0.20m × 0.015m, head 0.20m × 0.03m
- Cubes: small 0.01m blocks, low mass, easily pushed by wiper head
- World bounds: navigation constrained by room boundaries and fixtures
- Collision avoidance: active for all objects and fixtures during motion planning

Execution Timing:
- open_drawer: ~300 action steps (30 seconds at 10Hz control)
- pick_wiper: ~300 action steps (30 seconds)
- sweep: ~200 action steps (20 seconds)
- Total task: ~800 action steps (80 seconds) for Example 1

**Strategy Selection Guidelines:**
- For drawer opening: Use distance ≈ 0.7m, rotation ≈ -π for frontal approach
- For wiper picking: Use distance ≈ 0.7m, rotation ≈ -π for shelf 2 access
- For sweeping: Use distance ≈ 0.55-0.60m, rotation ≈ -π with slight adjustment for object distribution
- Multi-step tasks: Execute skills sequentially with state updates between skills
- Tool manipulation: Account for held object geometry in collision checking
- Trajectory planning: Use PyBullet IK and motion planning for collision-free paths
- Parameter sampling: Use sample_parameters() with different seeds for variation
- Object parameters: Include all entities (robot, wiper, drawer, all cubes) for complete scene awareness
