**Example 1: Pick Stick and Press Button**

Initial State:
- Robot: position (0.919, 0.405), angle -0.961 rad (-55°), arm length 0.1m, vacuum off
- Stick: position (2.915, 0.759), angle 0° (vertical), dimensions 0.05m × 1.25m
- Button0: position (1.687, 0.260), radius 0.05m (unpressed, red, on floor)

Goal: Press button0 to turn it green

Strategy: Pick up stick using gripper → Maneuver stick to contact button

High-Level Plan:
1. pick_stick_from_nothing(robot, stick, grasp_offset=[0.5, 0.5])
   - Navigate robot to stick position (2.915, 0.759)
   - Grasp offset [0.5, 0.5] indicates normalized grasp point on stick (center of stick)
   - Extend arm to reach stick's center
   - Activate vacuum (set to 1.0) when gripper overlaps stick
   - Stick becomes rigidly attached to gripper, moving with robot

2. stick_press_button_from_nothing(robot, stick, button0, approach=[0.0])
   - Maneuver robot with attached stick toward button0 at (1.687, 0.260)
   - Approach parameter [0.0] specifies strategy for positioning stick relative to button
   - Navigate base to position where stick geometry intersects with button
   - Contact triggers button press (button turns green)

Goal Reached:
- Button0 pressed (green) ✓
- Stick used as tool to extend reach ✓

Key Insights:
- Grasp offset [0.5, 0.5] represents normalized position along stick (0.5 = center)
- Vacuum must be > 0.5 to maintain grasp
- Stick rotates with robot, changing its sweep area
- Any part of stick contacting button triggers press

**Example 2: Direct Robot Press (No Stick Needed)**

Initial State:
- Robot: position (0.695, 0.315), angle 0.920 rad (53°), arm length 0.1m, vacuum off
- Stick: position (2.153, 0.652), angle 0° (not needed for this example)
- Button0: position (0.587, 0.731), radius 0.05m (unpressed, red, on floor)

Goal: Press button0

Strategy: Navigate robot base directly to button (skip stick pickup)

High-Level Plan:
1. robot_press_button_from_nothing(robot, button0, approach=[0.0])
   - Calculate direct path from robot position (0.695, 0.315) to button (0.587, 0.731)
   - Navigate robot base to overlap with button position
   - Approach parameter [0.0] indicates direct approach strategy
   - Robot base contact with button triggers press
   - Button turns green when robot base overlaps button circle

Goal Reached:
- Button0 pressed (green) ✓
- Efficient solution: no stick manipulation needed ✓

Key Insights:
- Floor-level buttons can be pressed by robot base directly
- This is more efficient than picking up stick when button is accessible
- Robot base radius 0.1m provides contact area for button pressing

**Example 3: Another Direct Press Scenario**

Initial State:
- Robot: position (2.073, 0.470), angle -1.373 rad (-79°), arm length 0.1m, vacuum off
- Stick: position (0.537, 0.710), angle 0°
- Button0: position (1.789, 0.131), radius 0.05m (unpressed, red, on floor)

Goal: Press button0

Strategy: Direct navigation to button (stick not needed)

High-Level Plan:
1. robot_press_button_from_nothing(robot, button0, approach=[0.0])
   - Navigate from (2.073, 0.470) to button at (1.789, 0.131)
   - Distance to button: ~0.38m (within direct reach)
   - Move robot base to contact button
   - Button pressed when base overlaps button

Goal Reached:
- Button0 pressed (green) ✓

Key Insights:
- Planner chooses direct press when button is on floor and accessible
- Stick pickup adds complexity, avoided when unnecessary

**Example 4: Distant Button Requires Stick**

Initial State:
- Robot: position (0.399, 0.322), angle -0.627 rad (-36°), arm length 0.1m, vacuum off
- Stick: position (3.138, 0.646), angle 0°, dimensions 0.05m × 1.25m
- Button0: position (3.139, 0.162), radius 0.05m (unpressed, red, on floor)

Goal: Press button0

Strategy: Pick stick → Use stick to reach distant button

High-Level Plan:
1. pick_stick_from_nothing(robot, stick, grasp_offset=[0.5, 0.5])
   - Navigate from (0.399, 0.322) to stick at (3.138, 0.646)
   - Distance to stick: ~2.8m (requires navigation)
   - Grasp stick at center point [0.5, 0.5]
   - Activate vacuum to attach stick to gripper

2. stick_press_button_from_nothing(robot, stick, button0, approach=[0.0])
   - Button0 is very close to stick's initial position (3.139, 0.162)
   - Maneuver robot with stick to position stick over button
   - Stick extends 0.625m from grasp point (half of 1.25m length)
   - Position robot so stick geometry overlaps button at (3.139, 0.162)
   - Contact triggers press

Goal Reached:
- Button0 pressed (green) ✓
- Stick provided necessary reach extension ✓

Key Insights:
- Button near stick's location suggests stick is optimal tool
- Stick length 1.25m provides 0.625m reach extension from robot gripper
- Rotating robot while holding stick sweeps stick through space

**General Operation Interpretation:**
- pick_stick_from_nothing(robot, stick, grasp_offset=[dx, dy]): 
  - Navigate to stick, extend arm, activate vacuum when overlapping
  - Grasp offset [0.5, 0.5] = center of stick (normalized coordinates)
  - Grasp offset [0.0, 0.0] = bottom of stick, [1.0, 1.0] = top
  
- stick_press_button_from_nothing(robot, stick, button, approach=[param]):
  - Maneuver robot with attached stick to contact button
  - Approach parameter guides positioning strategy
  - Success when any part of stick overlaps button circle
  
- robot_press_button_from_nothing(robot, button, approach=[param]):
  - Navigate robot base directly to button position
  - Success when robot base (radius 0.1m) overlaps button circle (radius 0.05m)
  - Efficient for floor-level buttons within reach

**Strategy Selection:**
- Use direct robot press when button is on floor and within ~3m direct navigation distance
- Use stick when button is on elevated table (y > world_height/2) or requires extended reach
- Stick provides ~1.25m tool length, with grasp typically at center (0.625m extension either direction)
