**Example 1: Two Cubes and One Box - Sequential Transport**

Initial State:
- Robot: base at (0.0, 0.0, 0.0), gripper open
- Table: center at (0.6, 0.0, 0.2), dimensions 0.4m × 0.8m × 0.4m, surface at z=0.4m
- Cube1: position (0.067, -0.673, 0.025) on ground, size 0.05m × 0.05m × 0.05m
- Cube0: position (0.879, -1.419, 0.025) on ground, size 0.05m × 0.05m × 0.05m
- Box0: position (0.355, -0.803, 0.1) on ground, dimensions 0.2m × 0.3m × 0.2m

Goal: Place all objects (cube1, cube0, box0) on the table (height > 0.3m, not held by robot)

Strategy: Prioritize objects by accessibility → Transport each sequentially → Place at different table locations

High-Level Plan:
1. pick(robot, cube1, grasp_offset=[0.55, 0.0])
   - Navigate robot to cube1 at (0.067, -0.673, 0.025)
   - Approach with base offset ensuring end effector can reach cube from position offset by (0.55, 0.0) relative to object
   - Lower gripper to z=0.025 (cube center height)
   - Close gripper to grasp cube1

2. place(robot, cube1, table, place_offset=[-0.10, -0.15])
   - Transport cube1 to table center (0.6, 0.0, 0.4)
   - Position cube at offset (-0.10, -0.15) from table center → final position (0.50, -0.15, 0.425)
   - Lower until cube contacts table surface
   - Open gripper to release cube1
   - Retreat to distance > 0.2m from cube

3. pick(robot, cube0, grasp_offset=[0.56, 0.2])
   - Navigate to cube0 at (0.879, -1.419, 0.025)
   - Approach with base offset (0.56, 0.2) for optimal reach angle
   - Lower and grasp cube0

4. place(robot, cube0, table, place_offset=[0.10, -0.15])
   - Transport cube0 to table
   - Place at offset (0.10, -0.15) from table center → final position (0.70, -0.15, 0.425)
   - Position next to cube1 but with separation to avoid collision
   - Release and retreat

5. pick(robot, box0, grasp_offset=[0.58, -0.2])
   - Navigate to box0 at (0.355, -0.803, 0.1)
   - Approach with base offset (0.58, -0.2)
   - Lower gripper to z=0.1 (box center height)
   - Grasp box from above/sides

6. place(robot, box0, table, place_offset=[0.00, 0.15])
   - Transport box0 to table
   - Place at offset (0.00, 0.15) from table center → final position (0.60, 0.15, 0.5)
   - Position box in available space away from cubes
   - Release and retreat

Goal Reached:
- Cube1 at (0.50, -0.15, 0.425): z=0.425m > 0.3m ✓
- Cube0 at (0.70, -0.15, 0.425): z=0.425m > 0.3m ✓
- Box0 at (0.60, 0.15, 0.5): z=0.5m > 0.3m ✓
- All objects on table, gripper open, not held ✓

Key Insights:
- Grasp offsets [dx, dy] determine robot base positioning relative to target object for optimal reachability
- Place offsets [dx, dy] specify object placement location relative to surface center
- Sequential transport: closest/easiest objects first, then farther objects
- Spatial planning: distribute objects across table surface to avoid collisions
- Table dimensions (0.4m × 0.8m) provide sufficient space for multiple objects

**Example 2: Two Cubes and One Box - Different Configuration**

Initial State:
- Robot: base at (0.0, 0.0, 0.0), gripper open
- Table: center at (0.6, 0.0, 0.2), surface at z=0.4m
- Cube0: position (-0.124, -0.684, 0.025) on ground, size 0.05m × 0.05m × 0.05m
- Cube1: position (-0.258, -1.148, 0.025) on ground, size 0.05m × 0.05m × 0.05m
- Box0: position (0.394, -1.115, 0.1) on ground, dimensions 0.2m × 0.3m × 0.2m

Goal: Place all objects on table

Strategy: Handle cubes first with consistent offsets → Transport box last

High-Level Plan:
1. pick(robot, cube0, grasp_offset=[0.55, 0.0])
   - Navigate to cube0 at (-0.124, -0.684, 0.025)
   - Standard grasp offset (0.55, 0.0) for straight approach
   - Grasp cube0

2. place(robot, cube0, table, place_offset=[-0.10, -0.15])
   - Transport to table
   - Place at offset (-0.10, -0.15) → position (0.50, -0.15, 0.425)
   - This is the same placement location as cube1 in Example 1 (reusable placement strategy)
   - Release and retreat

3. pick(robot, cube1, grasp_offset=[0.55, 0.0])
   - Navigate to cube1 at (-0.258, -1.148, 0.025)
   - Use same grasp offset (0.55, 0.0) as cube0 for consistency
   - Grasp cube1

4. place(robot, cube1, table, place_offset=[-0.10, 0.15])
   - Transport to table
   - Place at offset (-0.10, 0.15) → position (0.50, 0.15, 0.425)
   - Mirror y-position of cube0 placement (0.15 vs -0.15) for symmetric distribution
   - Release and retreat

5. pick(robot, box0, grasp_offset=[0.58, 0.0])
   - Navigate to box0 at (0.394, -1.115, 0.1)
   - Slightly larger grasp offset (0.58 vs 0.55) to account for box size
   - Grasp box

6. place(robot, box0, table, place_offset=[0.10, 0.00])
   - Transport to table
   - Place at offset (0.10, 0.00) → position (0.70, 0.00, 0.5)
   - Center box in y-direction, offset in x to avoid cubes
   - Release and retreat

Goal Reached:
- Cube0 at (0.50, -0.15, 0.425): z=0.425m > 0.3m ✓
- Cube1 at (0.50, 0.15, 0.425): z=0.425m > 0.3m ✓
- Box0 at (0.70, 0.00, 0.5): z=0.5m > 0.3m ✓

Key Insights:
- Consistent grasp offsets: similar objects (cubes) use same offset (0.55, 0.0)
- Symmetric placement: cubes placed with mirrored y-offsets (-0.15, 0.15) for spatial distribution
- Box placement: centered at y=0.0 with positive x-offset to use remaining table space
- Offset patterns can be reused across different initial configurations
- Small differences in grasp offset (0.55 vs 0.58) accommodate different object sizes

**General Offset Interpretation:**
- pick(grasp_offset=[dx, dy]): robot base positions at (object_x - dx, object_y - dy) to reach object with end effector
- place(place_offset=[dx, dy]): object final position is (surface_x + dx, surface_y + dy, surface_z + object_height/2)
- Positive grasp offset dx: robot approaches from behind object (negative x direction)
- Grasp offset magnitude (0.55-0.58): typical reach distance for this robot configuration
- Place offsets distribute objects: negative x for left side, positive x for right side, varied y for front/back
