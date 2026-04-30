# Changelog

## [Unreleased] — 2026-04-30 (continued 5)

### Blocks — Abstraction (use_skill params)

- **`use_skill` param values use `=`** — collapsed view now shows `name = value` instead of `name: value`.
- **Type system revised** — `string` removed; `float` added (displayed as "floating point" in `define_skill` dropdown, "float" in compact collapsed view). `int` now displays as "integer" in the expanded dropdown.
- **`define_skill` param count preserved on resize** — increasing the count keeps existing param names/types; only new rows get fresh defaults.
- **`use_skill` param inputs** — int/float params use `FieldTextInput` defaulting to `"NULL"` (user types a number or `"NULL"`). Color params use `FieldColorSwatch` with a `FieldLabel` NULL indicator.
- **NULL state** — new params default to `"NULL"` in the text box. Typing a real value clears the red. Color params show a `" NULL"` label next to the swatch when stale; picking a color clears it.
- **Red name on param count mismatch** — `use_skill` collapsed skill name goes red when the referenced skill exists but param count differs.
- **Red name on invalid skill** — skill name goes red when no matching `define_skill` exists.
- **Imprecise float→int** — if a float value (e.g. `1.5`) is mapped to an int param, it is not silently rounded: the original value is preserved in `impreciseSaved_` and shown in red. Reverting to float restores the original value.
- **Stale kind mismatch** — incompatible type changes (num↔color) set `staleSaved_` for that index; collapsed view shows the current value (or `NULL` for color) in red. Picking a new color clears the stale flag.
- **Red values survive type revert** — snapshotting prefers `impreciseSaved_` over the zeroed field so the original value is restored when the type reverts.
- **Live `define_skill` propagation** — any `define_skill` field change calls `updateParamInputs_()` on all `use_skill` blocks (not just a re-collapse), keeping param inputs, null states, and validity in sync.
- **`isUseSkillValid` extended** — also checks `impreciseSaved_`, `staleSaved_`, and `"NULL"` text values before enabling a `use_skill` block.

### Blockly Workspace

- **Workspace persistence across hot reloads** — workspace state is serialized to `localStorage` 400 ms after any change and restored on mount. Zoom and scroll are always reset to 1.5× centred on origin after restore.

---

## [Unreleased] — 2026-04-30 (continued 4)

### Blocks — Abstraction

- **`define_skill` block** — new navy-blue (`#1e3a8a`) block in a new "Abstraction" toolbox category (after Pen). Has no top notch (sequence head only). Fields: skill name (text input), param count (0–5 dropdown). Sequences connected below are enabled just like `Start`.
- **`define_skill` param rows** — selecting a param count dynamically adds/removes bullet rows (`• name : type`) with editable name and a type dropdown (int / string / color). Uses `saveExtraState`/`loadExtraState` for correct serialization of dynamic inputs.
- **`define_skill` custom collapse** — instead of Blockly's single-line collapse, uses input visibility toggling. Collapsed view shows Python-style long syntax: `def name(` / `    param: type,` / `)`. Zero-param skills collapse to a single line `def name()`. The skill name is underlined via a custom `FieldLabelUnderline` field subclass.
- **`use_skill` block** — teal (`#14b8a6`) block with a dynamic dropdown listing all `define_skill` names currently on the workspace. Collapses to `name(` / `)` with the skill name underlined. Skill name forwarded in `getProgram` output as `entry.skill`.
- **Skill enable/disable propagation** — `updateEnabledStates` disables `use_skill` blocks whose referenced name doesn't match any current `define_skill` name. Re-runs on `BLOCK_CHANGE` when a `define_skill` NAME field is edited.
- **`CUSTOM_COLLAPSE` set** — `BlocklyWorkspace` routes `define_skill` and `use_skill` through `customCollapse_`/`customExpand_` instead of `setCollapsed`, both on selection change and `BLOCK_CREATE`.

### Color picker

- **Picker anchors to block only** — color picker now opens directly below the `set_pen_color` block itself using `:scope > .blocklyPath` bounding rect, not the whole connected sequence.

---

## [Unreleased] — 2026-04-30 (continued 3)

### Blockly Workspace

- **"Duplicate from here" context menu** — right-clicking any non-`start` block shows a "Duplicate from here" option that copies the block and its entire following sequence, places the copy 40 px offset, and collapses all duplicated blocks.
- **Remove "Expand block" context menu item** — `blockCollapseExpand` unregistered on workspace init (joins `blockCollapse` and `blockDisable`).
- **One `start` block enforced** — if a second `start` block is dragged from the flyout, TidyBot immediately says "You can only start once!" and the duplicate is disposed when the drag is released.

### UI

- **TidyBot event dispatch** — `BlocklyWorkspace` now dispatches a `message` event (via Svelte `createEventDispatcher`) so workspace-level interactions can trigger TidyBot messages. `App.svelte` handles it with `on:message={e => tamaSay(e.detail)}`.
- **Favicon** — browser tab now shows the TidyBot pixel-art sprite (`public/favicon.svg`) — a faithful SVG recreation of the CSS box-shadow pixel art with the retro dark background.

---

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
