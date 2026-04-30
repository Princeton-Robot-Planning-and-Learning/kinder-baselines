# Changelog

## [Unreleased] — 2026-04-30 (continued 2)

### Blocks

- **`start` block** — new light-green (`#4ade80`) block in a new "Program" category (placed first in the toolbox). Has only a bottom notch (next statement only); no top notch. Programs must begin with this block.
- **Auto-disable unconnected blocks** — a `BLOCK_MOVE` / `BLOCK_CREATE` / `BLOCK_DELETE` change listener calls `updateEnabledStates()` on every structural change. It walks every `start` block's chain to build a reachable set, enables those blocks, and disables all others. Disconnected blocks appear greyed-out; reconnecting re-enables them. `getProgram()` skips `start`-type blocks so they are not sent to the executor.
- **`set_pen_color` dynamic background** — the block's background colour updates to the current R/G/B value via field validators so the block acts as a live colour swatch. Collapsed blocks show their pen colour as the block background.
- **`set_pen_color` no longer puts pen down** — `set_pen_color` now only changes the colour; students must use an explicit `pen_down` block to start drawing.
- **Honeycomb color picker revised** — swapping removed; all seven hex tiles stay in fixed positions. The currently-selected colour gets a white glow highlight. The selected tile is rendered last in SVG order so its glow is never clipped by neighbours.

### Blockly Workspace

- **Double-click to reset view** — double-clicking the workspace background resets zoom to 1.5× and re-centres on the origin. Double-clicks on blocks are ignored.
- **Larger right-click menu** — context menu font bumped to 20 px with 10 px vertical padding per item via injected CSS (`.blocklyContextMenu`, `.blocklyMenuItem`, `.blocklyMenuItemContent`).

---

## [Unreleased] — 2026-04-30 (continued)

### Executor

- **Y-axis depth fix** — positive UI Y now moves the robot *away* from the camera (towards the back wall) in the 3D view. Camera is positioned on the +X side, so robot X is negated when converting from UI Y: `target_x = -UI_Y`. The same negation applies to `move_base_by`.
- **Disabled block skip** — `getProgram()` now calls `block.isEnabled()` and skips disabled (greyed-out) blocks while continuing to execute the rest of the stack.
- **Implicit pen-up event** — if the program ends with the pen still down, an implicit `pen_up` event is recorded so the ○ marker always appears at the end of every stroke.

### Canvas / Trail

- **Pen event markers (physics RHR)** — `drawPenMarkers(canvas, events)` draws × (pen entering page) at pen-down positions and ○ (pen leaving page) at pen-up positions, both in the current pen colour. When pen-up and pen-down occur at the same location the two symbols naturally overlay to form ⊗. Events are collected in `_PenState.events` and forwarded through server → App → OutputPanel.
- **Removed trail start/end dots** — the filled start dot (r=5) and end dot (r=7) previously drawn by `drawCanvasTrail` have been removed; the × and ○ pen markers serve this purpose.
- **Trail x-axis negation** — `drawCanvasTrail` and `drawPenMarkers` negate the robot-X coordinate (`-seg.x1`, `-ev.x`) before passing to `worldToCanvas`, keeping the canvas display consistent with the new UI Y = −robot X convention.

### Challenges

- **X-coordinate negation** — all challenge trail waypoints have their first coordinate (robot X) negated to match the new executor convention. Shapes display identically on the canvas; scoring against student trails remains correct.

### Blocks / Color Picker

- **Honeycomb color picker** — clicking an already-selected `set_pen_color` block opens a retro 7-hex honeycomb picker (`ColorPicker.svelte`). Seven preset colors (red, orange, yellow, green, cyan, blue, violet) are arranged with the current color in the center and the remaining six surrounding it. Clicking a surrounding hex swaps it with the center and immediately updates the block's R/G/B fields. Clicking the center or anywhere outside closes the picker. The picker is positioned below the block using `getBoundingClientRect` and rendered via `position:fixed` so it escapes the workspace scroll container. A `click` listener (not `pointerdown`) on the blockly div gates opening so block drags don't accidentally trigger it.

---

## [Unreleased] — 2026-04-30

### Blockly Workspace

- **Default zoom & centering** — workspace starts at scale 1.5, centred on origin (0, 0).
- **Fixed content bounds** — replaced Blockly's dynamic `MetricsManager` with `FixedSizeMetricsManager` that returns a constant 5000×5000 workspace-unit area centred at the origin. Scrollbar thumb sizes are now stable regardless of how many blocks are placed.
- **Zoom-to-cursor** — `Ctrl+scroll` (and `Ctrl+`/`Ctrl-`) zooms around the cursor position. The large symmetric content area prevents scroll clamping that was shifting the focal point.
- **Grid dot colour** — workspace grid dots changed from near-black `#1a0a3d` to visible purple `#5b21b6`.

### Blocks

- **`move_base_by` block** — new relative-movement block ("Move base by x _ y _") in the Movement category. Uses ±4 range for DX/DY. Rendered in a lighter purple (`#a88fe0`) to distinguish it from the absolute `move_base_to_target` block.
- **Coordinate convention fix** — UI now uses standard convention: **X = horizontal axis, Y = vertical axis**. Previously X was the robot's forward axis (vertical on screen) and Y was the side axis (horizontal). The executor swaps back to robot coordinates internally; trail drawing is unaffected.

### Target Canvas Interaction

- **Click to populate `move_base_to_target`** — clicking anywhere on the target canvas while a `move_base_to_target` block was last selected sets the block's X/Y fields to the snapped world coordinate. Works even though clicking the canvas deselects the Blockly block (last-selected block is tracked via a Blockly `selected` event listener).
- **Drag to populate `move_base_by`** — mousedown + drag on the target canvas draws a purple vector arrow with start dot and arrowhead. On release, DX/DY are set on the last-selected `move_base_by` block. Short press (<5 px movement) still triggers the click-to-populate path.
- **Hover crosshair** — a transparent overlay canvas shows a dashed crosshair and glowing dot snapped to 0.1 precision while hovering over the target canvas.

### Output Panel

- **Frame navigation** — `<` and `>` buttons flank the 3D view for stepping through frames one at a time. Buttons are purple when active, near-black when at the limit. Auto-play on Run still animates all frames; navigation is available before and after playback.
- **Responsive canvas sizing** — both canvases (Target, Your Drawing) shrink proportionally when their combined width would exceed the 3D view image width.
- **Label alignment** — "// TARGET" gains `padding-top` equal to the height difference when "// YOUR DRAWING" wraps to two lines, keeping the canvases vertically aligned. Uses `scrollHeight` (not `offsetHeight`) to avoid a circular measurement bug. Padding is also applied on initial mount via `requestAnimationFrame`.
- **Canvas centering** — layout uses CSS Grid (`auto | auto | auto` columns) so the `<`/`>` buttons occupy the outer columns and the image, frame-info, and canvas row all sit in the same middle column. This guarantees the canvas row is centred under the image regardless of button width.
- **Label text-align** — all `// PANEL LABEL` text is centre-aligned (affects the two-line "// YOUR DRAWING" case).
- **Trail/canvas dots** — start dot (radius 5, blur 14) and end dot (radius 7, blur 18) added to trail drawings.

### UX

- **Escape to silence TamaBot** — pressing `Escape` immediately hides the TamaBot speech bubble and cancels the auto-dismiss timer.
