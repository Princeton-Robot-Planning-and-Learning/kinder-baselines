**Example 1: Hook Manipulation to Push Target Block (With Obstructions)**

Initial State:
- Robot: position (0.362, 0.394), angle 0.306 rad (≈17.5°), arm length 0.48m (fully extended), gripper open (gap 0.32m)
- Hook: position (2.706, 1.013), angle -0.142 rad (≈-8.1°), L-shaped tool with dimensions width=0.107m, length_side1=1.4m, length_side2=0.583m, mass 1.0kg
- Target block: position (2.297, 2.833), angle -1.842 rad (≈-105.5°), dimensions 0.405m × 0.405m (square), mass 1.0kg, located in upper region
- Middle wall (goal surface): position (1.75, 1.75), black horizontal barrier spanning full world width (3.5m × 0.05m)
- Obstruction0: position (2.167, 2.499), angle -2.548 rad, dimensions 0.173m × 0.173m
- Obstruction1: position (1.854, 3.141), angle -0.859 rad, dimensions 0.224m × 0.192m
- Obstruction2: position (2.570, 3.203), angle -0.279 rad, dimensions 0.433m × 0.282m
- Obstruction3: position (2.813, 2.773), angle 0.788 rad, dimensions 0.228m × 0.340m
- Obstruction4: position (3.293, 2.627), angle 0.696 rad, dimensions 0.355m × 0.248m
- World boundaries: 3.5m × 3.5m, no gravity (gravity_y = 0.0), damping = 0.01

Goal: Use the L-shaped hook as a tool to push/pull the target block from the upper region (y ≈ 2.833) down to the middle wall at center (y = 1.75), navigating through clustered obstructions

Strategy: Grasp hook → Position hook near target → Use hook geometry to push/pull target downward to goal

High-Level Plan:
1. GraspHook(robot, hook, grasp_param=[0.621])
   - Navigate robot from starting position (0.362, 0.394) toward hook at (2.706, 1.013)
   - grasp_param 0.621: normalized grasp point along hook's longer arm (length_side1 = 1.4m)
     - 0.0 = grasp at corner vertex, 1.0 = grasp at far end of side1
     - 0.621 ≈ 0.87m from corner along the 1.4m arm (strategic balance point)
   - Approach hook, extend arm if needed, align gripper with hook geometry
   - Close gripper fingers (reduce finger_gap from 0.32m to match hook width 0.107m + finger considerations)
   - Hook becomes held, rigidly attached to robot gripper through physics constraints
   - Hook orientation relative to gripper is maintained during manipulation
   - Executed in 36 action steps

2. PreHook(robot, hook, target_block, position_params=[2.034, -0.040, -0.013])
   - Maneuver robot with held hook to position hook strategically near target block
   - position_params = [distance, offset_x, offset_y]:
     - distance 2.034: desired radial distance from target block center to robot base
     - offset_x -0.040: fine-tuning horizontal offset for hook positioning
     - offset_y -0.013: fine-tuning vertical offset for hook positioning
   - Goal: Position hook's L-shape to engage with target block geometry
   - Navigate around obstructions (5 obstacles scattered near target)
   - Hook's long arm (1.4m) provides reach to contact target from strategic angle
   - Robot maintains grasp throughout movement (gripper stays closed)
   - Damping (0.01) provides slight resistance to motion, requiring controlled movement
   - Executed in 39 action steps

3. HookDown(robot, hook, target_block, params=[0.0])
   - Execute pulling/pushing motion to drag target block downward toward middle wall
   - params 0.0: pulling strategy parameter (specific approach for downward motion)
   - Robot moves with hook, using hook's geometry as lever to apply force to target
   - L-shape hook design allows catching/pushing target block effectively
   - No gravity, so motion relies purely on robot force transmitted through hook
   - Target block must navigate past obstructions during descent
   - Physics simulation handles:
     - Contact forces between hook and target block
     - Collisions between target and obstructions (elastic collisions with friction)
     - Damping effects on all dynamic objects
   - Continue motion until target block intersects middle wall geometry
   - Success condition: target block geometry overlaps with middle wall (goal surface)
   - Executed in 25 action steps

Goal Reached:
- Target block moved from upper region (y ≈ 2.833) to middle wall (y = 1.75) ✓
- Hook used as extended tool to manipulate target indirectly ✓
- Navigated through 5 obstructions successfully ✓
- Hand still holding hook (HandEmpty predicate false, HoldingHook true) ✓
- Total actions: 36 + 39 + 25 = 100 steps

Key Insights:
- Hook is an L-shaped tool providing extended reach (1.4m + 0.583m arms)
- No gravity means objects don't fall naturally; robot must actively push/pull
- Damping (0.01) provides slight drag on all moving objects
- Grasp point selection (0.621) balances control and reach for hook manipulation
- PreHook positioning is critical for effective force application
- Target must intersect middle wall geometry to satisfy goal condition
- Obstructions add complexity but don't fundamentally change strategy
- Action space: [dx, dy, dtheta, darm, dgripper] with ranges ±0.05m, ±0.05m, ±π/48 rad, ±0.1m, ±0.02m per step

**Example 2: Simpler Hook Task (Fewer Obstructions)**

Initial State:
- Robot: position (0.35, 0.38), angle 0.5 rad, arm length 0.48m, gripper open
- Hook: position (2.50, 1.20), angle 0.0 rad, L-shaped with standard dimensions (width=0.107m, sides 1.4m × 0.583m)
- Target block: position (2.80, 2.60), angle 0.0 rad, dimensions 0.35m × 0.35m, mass 1.0kg
- Middle wall: position (1.75, 1.75), spanning full width (goal surface)
- Obstruction0: position (2.60, 2.40), dimensions 0.20m × 0.25m
- Obstruction1: position (2.95, 2.30), dimensions 0.18m × 0.22m
- World: 3.5m × 3.5m, no gravity, damping 0.01

Goal: Pull target block to middle wall using hook tool

Strategy: Grasp hook → Position hook above/beside target → Pull target down to goal

High-Level Plan:
1. GraspHook(robot, hook, grasp_param=[0.55])
   - Navigate from (0.35, 0.38) to hook at (2.50, 1.20)
   - Grasp at point 0.55 along hook's main arm (≈0.77m from corner)
   - Close gripper around hook width (0.107m)
   - Hook held securely by robot

2. PreHook(robot, hook, target_block, position_params=[1.80, 0.05, -0.08])
   - Position robot with hook near target at (2.80, 2.60)
   - distance 1.80: closer approach for better control
   - offset_x 0.05, offset_y -0.08: position hook to catch target from side
   - Align hook's L-shape to engage target block effectively
   - Navigate around 2 obstructions during approach

3. HookDown(robot, hook, target_block, params=[0.0])
   - Execute downward pulling motion with hook
   - Use hook's long arm to maintain contact with target
   - Pull target from y ≈ 2.60 down to middle wall at y = 1.75
   - Distance traveled: ~0.85m downward
   - Obstructions passively affected by target motion (collisions handled by physics)
   - Goal achieved when target intersects middle wall

Goal Reached:
- Target on middle wall ✓
- Hook used effectively as pulling tool ✓
- Fewer obstructions meant faster execution ✓

Key Insights:
- With fewer obstructions, PreHook positioning is simpler
- Grasp point (0.55) closer to center provides balanced control
- Hook's length (1.4m main arm) allows reaching target from safe distance
- Damping slows objects gradually, preventing uncontrolled motion
