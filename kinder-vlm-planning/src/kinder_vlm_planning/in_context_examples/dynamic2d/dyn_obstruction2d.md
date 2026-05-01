**Example 1: Simple Pick and Place (One Obstruction, Not Blocking Target)**

Initial State:
- Target surface: position (1.798, 0.050), angle 0.0 rad, dimensions 0.325m × 0.10m (width × height), purple color (0.502, 0.0, 0.502)
- Target block: position (2.982, 0.228), angle 0.0 rad, dimensions 0.267m × 0.258m (width × height), mass 1.0kg, purple color
- Obstruction0: position (1.641, 0.262), angle 0.0 rad, dimensions 0.302m × 0.327m, mass 1.0kg, red color (0.75, 0.1, 0.1)
  - Note: Obstruction is away from target surface (not blocking it)
- Robot: position (0.515, 0.416), angle -0.242 rad (≈-13.9°), arm length 0.48m (fully extended), gripper open (gap 0.32m)
- Table: static surface at y = 0.05m (height 0.1m)
- World: 2.618m × 2.0m (golden ratio width), gravity enabled

Goal: Place target block completely on the target surface (both purple, must align)

Strategy: Since obstruction is not blocking target surface, directly pick target block and place on target surface

High-Level Plan:
1. PickFromTable(robot, target_block, grasp_params=[0.0103, 0.598, 0.302])
   - Navigate robot from starting position (0.515, 0.416) toward target block at (2.982, 0.228)
   - Distance to travel: ~2.5m horizontally
   - grasp_params = [grasp_ratio, side, arm_length]:
     - grasp_ratio 0.0103: grasp very near the edge of the top side of the block (almost at corner)
     - side 0.598: approach from slightly right of center (0.5 = center, 0.598 indicates slight bias)
     - arm_length 0.302: extend arm from current 0.48m (fully extended) to 0.302m for grasping approach
   - Navigate around obstruction at (1.641, 0.262) during approach
   - Rotate robot to align gripper with block's top edge
   - Extend/retract arm to arm_length 0.302m for optimal grasp position
   - Close gripper fingers: reduce finger_gap from 0.32m to accommodate block width 0.267m
     - Desired finger_gap ≈ 0.267m + 0.2m (finger_width) - 0.175m ≈ 0.292m (slight compression for grasp)
   - Block becomes held (held=1), rigidly attached to robot gripper through PyMunk revolute joint constraints
   - Gripper maintains grasp through friction (1.0) and collision detection
   - Executed in 60 action steps

2. PlaceOnTarget(robot, target_block, target_surface, placement_param=[0.25])
   - Navigate robot with held block from pick location toward target surface at (1.798, 0.050)
   - Target block currently at ~(2.982, 0.228), target surface at (1.798, 0.050)
   - placement_param 0.25: normalized position along target surface width
     - Surface width 0.325m, so 0.25 × 0.325m ≈ 0.081m from left edge
     - Target x-position: 1.798 - 0.325/2 + 0.081 ≈ 1.636m
   - Navigate robot to position where held block will be placed at x ≈ 1.636m on target surface
   - Avoid collision with obstruction0 at (1.641, 0.262) during approach
   - Lower arm (if needed) and open gripper to release block
   - Increase finger_gap from ~0.292m back toward 0.32m to release
   - Physics simulation: block drops onto target surface under gravity
   - Goal condition verified: target block's bottom vertices within target surface bounds
   - Target block (width 0.267m) fits on target surface (width 0.325m) with margin
   - Executed in 28 action steps

Goal Reached:
- Target block on target surface (OnTarget predicate satisfied) ✓
- Hand empty (HandEmpty predicate satisfied, held=0) ✓
- Total actions: 60 + 28 = 88 steps
- Obstruction0 remains on table, not interfering with goal ✓

Key Insights:
- Grasp ratio very small (0.0103) indicates precision grasp near edge of block
- Side parameter (0.598) provides slight approach bias for better alignment
- Arm length parameter (0.302) is less than full extension (0.48m), indicating retraction for controlled grasp
- Placement parameter (0.25) positions block near left quarter of target surface for stable placement
- Physics ensures block settles properly on surface under gravity
- Obstruction present but not blocking target, so no clearing needed
- Color information helps identify target objects (both purple: target_block and target_surface)

**Example 2: Obstruction Clearance Required (One Obstruction Blocking Target)**

Initial State:
- Robot: position (0.50, 0.50), angle 0.0 rad, arm length 0.24m (retracted), gripper open (gap 0.32m)
- Target surface: position (1.80, 0.05), angle 0.0 rad, dimensions 0.30m × 0.10m, purple
- Target block: position (1.20, 0.35), angle 0.0 rad, dimensions 0.20m × 0.40m (width × height), mass 1.0kg, purple
- Obstruction0: position (1.80, 0.25), angle 0.0 rad, dimensions 0.28m × 0.35m, mass 1.0kg, red (blocking target surface)
- Table: y = 0.05m (height 0.1m)

Goal: Place target block on obstructed target surface

Strategy: Clear obstruction first, then pick and place target block

High-Level Plan:
1. PickFromTable(robot, obstruction0, grasp_params=[0.0, 0.65, 0.35])
   - Navigate from (0.50, 0.50) to obstruction at (1.80, 0.25)
   - grasp_ratio 0.0: grasp at edge of obstruction
   - side 0.65: approach with bias for better angle
   - arm_length 0.35: extend to 0.35m for grasping
   - Close gripper to secure obstruction (width 0.28m)
   - Obstruction becomes held

2. PlaceOffTarget(robot, obstruction0, clear_location_param=[0.8])
   - Transport obstruction away from target surface
   - clear_location_param 0.8: identifies collision-free location
   - Navigate to position (e.g., x > 2.2m) clear of target surface
   - Open gripper to release obstruction onto table
   - Verify obstruction OnTable but not OnTarget

3. PickFromTable(robot, target_block, grasp_params=[0.0, 0.6, 0.32])
   - Navigate to target block at (1.20, 0.35)
   - grasp_ratio 0.0: edge grasp
   - side 0.6: centered approach
   - arm_length 0.32: appropriate extension for block at this height
   - Close gripper around block (width 0.20m)
   - Target block held

4. PlaceOnTarget(robot, target_block, target_surface, placement_param=[0.5])
   - Navigate to now-clear target surface at (1.80, 0.05)
   - placement_param 0.5: center placement on surface
   - Position robot so block aligns with center of target surface
   - Release block onto surface
   - Block drops and settles on target surface

Goal Reached:
- Obstruction cleared (OnTable, not OnTarget) ✓
- Target block on target surface (OnTarget satisfied) ✓
- Hand empty ✓

Key Insights:
- Obstruction blocking requires 4 skill executions (2 × pick-place pairs)
- Clear location parameter must ensure obstruction doesn't re-block target
- Sequential manipulation requires careful state management (HandEmpty between picks)
- Each grasp adapts parameters to object dimensions and positions
