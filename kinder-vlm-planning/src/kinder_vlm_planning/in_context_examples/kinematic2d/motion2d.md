**Example 1: Open Space (No Obstacles)**

Initial State:
- Robot: position (0.3, 1.25, 0.0), base radius 0.1m
- Target Region: center (2.25, 1.25), size 0.25m × 0.25m (rectangle from 2.125 to 2.375 in both x and y)
- World: 2.5m × 2.5m, no obstacles

Goal: Robot base position must be inside target region

Strategy: Navigate directly to target region using collision-free motion planning

High-Level Plan:
1. move_to_tgt_from_no_passage(robot, target_region, params=[0.5, 0.5, 0.5])
   - Navigate robot directly to target region center
   - params=[0.5, 0.5, 0.5] specifies:
     - rel_x=0.5: target x-position at center of target region (50% across width)
     - rel_y=0.5: target y-position at center of target region (50% across height)
     - rel_theta=0.5: normalized angle (0.5 → 0° orientation after denormalization)
   - Motion planning computes collision-free path from (0.3, 1.25) to (2.25, 1.25)
   - No obstacles present, so direct path is optimal
   - Controller executes waypoint sequence to reach target

Goal Reached:
- Robot position (2.25, 1.25) is inside target region [2.125, 2.375] × [2.125, 2.375] ✓
- Efficient direct navigation with no obstacle avoidance needed ✓

Key Insights:
- params=[rel_x, rel_y, rel_theta] are normalized coordinates (0.0 to 1.0)
- rel_x, rel_y specify target position within rectangular target region
- rel_theta converts to absolute angle: abs_theta = rel_theta * 2π - π
- Motion planning ensures collision-free path

**Example 2: Single Narrow Passage**

Initial State:
- Robot: position (0.3, 1.25, 0.0), base radius 0.1m
- Target Region: center (2.25, 1.25), size 0.25m × 0.25m
- Obstacles: Two rectangles forming vertical wall at x=1.25 with passage opening
  - obstacle_bottom: center (1.25, 0.5), dimensions (0.01m × 1.0m), covers y=0.0 to y=1.0
  - obstacle_top: center (1.25, 2.0), dimensions (0.01m × 1.0m), covers y=1.5 to y=2.5
  - Passage opening: y=1.0 to y=1.5 (height 0.5m)

Goal: Navigate through passage to reach target region

Strategy: Use passage-aware motion planning to navigate through opening

High-Level Plan:
1. move_to_tgt_from_passage(robot, target_region, obstacle_bottom, obstacle_top, params=[0.5, 0.5, 0.5])
   - Navigate from starting position through passage to target region
   - params=[0.5, 0.5, 0.5] specifies center of target region
   - Motion planner considers obstacle_bottom and obstacle_top to identify passage
   - Computes path that passes through opening at y≈1.25 (center of 1.0-1.5 range)
   - Robot diameter (0.2m) fits through passage opening (0.5m tall) with clearance
   - Controller executes waypoints: approach passage → pass through → reach target

Goal Reached:
- Robot successfully navigated through narrow passage ✓
- Robot position inside target region ✓
- Collision-free path maintained throughout ✓

Key Insights:
- Passage-aware controller identifies opening between obstacles
- Robot base radius (0.1m) requires passage height > 0.2m for safe navigation
- Motion planning automatically centers robot in passage for maximum clearance
- Single skill call handles entire navigation sequence

**Example 3: Multiple Passages - Sequential Navigation**

Initial State:
- Robot: position (0.3, 1.25, 0.0), base radius 0.1m
- Target Region: center (2.25, 2.0), size 0.25m × 0.25m
- Obstacles: Six rectangles forming three vertical walls with passages
  - Wall 1 at x=0.7: obstacles at y=[0.0, 0.5] and y=[1.0, 2.5], passage y=0.5 to y=1.0
  - Wall 2 at x=1.25: obstacles at y=[0.0, 1.5] and y=[2.0, 2.5], passage y=1.5 to y=2.0
  - Wall 3 at x=1.8: obstacles at y=[0.0, 1.8] and y=[2.3, 2.5], passage y=1.8 to y=2.3

Goal: Navigate through all three passages to reach target at (2.25, 2.0)

Strategy: Chain passage navigation skills sequentially

High-Level Plan:
1. move_to_passage_from_no_passage(robot, wall1_bottom, wall1_top, params=[0.5, 0.5, 0.5])
   - Navigate from starting position (0.3, 1.25) to first passage
   - params specify position within passage opening (center at y=0.75)
   - Motion planner computes path to passage entrance

2. move_to_passage_from_passage(robot, wall2_bottom, wall2_top, wall1_bottom, wall1_top, params=[0.5, 0.5, 0.5])
   - Navigate from first passage to second passage
   - Current position: past wall 1, approaching wall 2
   - Target: passage 2 center at y=1.75 (between 1.5 and 2.0)
   - Motion planning accounts for both current and target passage constraints

3. move_to_tgt_from_passage(robot, target_region, wall3_bottom, wall3_top, params=[0.5, 0.8, 0.5])
   - Navigate from second passage through third passage to target
   - params=[0.5, 0.8, 0.5]: target position at (50% width, 80% height, 0° angle)
   - Final position: (2.25, 2.0) center of target region
   - Motion planner handles third passage automatically en route to target

Goal Reached:
- Robot successfully navigated through 3 passages ✓
- Robot position inside target region ✓
- Optimal path selected based on passage alignments ✓

Key Insights:
- Multiple passages handled by chaining passage navigation skills
- Each skill considers obstacles to compute collision-free waypoints
- Params allow fine-tuning target position within passage or region
- Motion planning optimizes path through available openings

**Example 4: Passage with Non-Centered Target**

Initial State:
- Robot: position (0.3, 0.5, 0.0), base radius 0.1m
- Target Region: center (2.25, 2.0), size 0.25m × 0.25m
- Obstacles: Two rectangles forming wall at x=1.25 with passage
  - obstacle_bottom: covers y=0.0 to y=1.5
  - obstacle_top: covers y=2.0 to y=2.5
  - Passage opening: y=1.5 to y=2.0 (height 0.5m)

Goal: Navigate to target near top edge of target region

Strategy: Use parameterized position within target region

High-Level Plan:
1. move_to_tgt_from_passage(robot, target_region, obstacle_bottom, obstacle_top, params=[0.5, 0.8, 0.25])
   - Navigate through passage to specific position within target region
   - params=[0.5, 0.8, 0.25] specifies:
     - rel_x=0.5: center of target region width (x ≈ 2.25)
     - rel_y=0.8: 80% up the target region height (y ≈ 2.15)
     - rel_theta=0.25: normalized angle (-45° after denormalization)
   - Motion planner:
     - Computes path from (0.3, 0.5) to passage at y≈1.75
     - Routes through passage opening (y=1.5 to y=2.0)
     - Continues to target position (2.25, 2.15) with orientation -45°
   - Controller executes waypoints with collision avoidance

Goal Reached:
- Robot at (2.25, 2.15) inside target region ✓
- Precise positioning achieved via parameterization ✓
- Efficient path through available passage ✓

Key Insights:
- Non-centered params (e.g., 0.8 vs 0.5) allow precise positioning
- Useful for tasks requiring specific target region locations
- Motion planning handles geometry constraints automatically

**General Parameterized Skill Interpretation:**

Skills:
- move_to_tgt_from_no_passage(robot, target, params=[rel_x, rel_y, rel_theta]):
  - Direct navigation when no obstacles block path to target
  - Params specify normalized target position and orientation

- move_to_tgt_from_passage(robot, target, obstacle1, obstacle2, params=[rel_x, rel_y, rel_theta]):
  - Navigate through passage defined by obstacle1/obstacle2 to reach target
  - Planner ensures path passes through opening between obstacles

- move_to_passage_from_no_passage(robot, obstacle1, obstacle2, params=[rel_x, rel_y, rel_theta]):
  - Navigate from open space to passage entrance
  - Params specify position within passage opening

- move_to_passage_from_passage(robot, obstacle1, obstacle2, obstacle3, obstacle4, params=[rel_x, rel_y, rel_theta]):
  - Navigate from one passage to another
  - obstacle1/2 define target passage, obstacle3/4 define current passage context

Parameters:
- rel_x ∈ [0, 1]: normalized x-position within target region or passage
  - 0.0 = left edge, 0.5 = center, 1.0 = right edge
- rel_y ∈ [0, 1]: normalized y-position within target region or passage
  - 0.0 = bottom edge, 0.5 = center, 1.0 = top edge
- rel_theta ∈ [0, 1]: normalized orientation
  - Converts to absolute: abs_theta = rel_theta × 2π - π
  - 0.0 → -π, 0.5 → 0, 1.0 → π

Motion Planning:
- Each skill uses RRT-based motion planning for collision-free paths
- Plans sequences of SE2 poses (x, y, theta) as waypoints
- Controller tracks waypoints using action space [dx, dy, dtheta, darm, vac]
- Arm (darm) and vacuum (vac) not used in this environment (stay at 0)
