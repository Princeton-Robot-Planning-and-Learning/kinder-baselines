<script>
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher();
  import * as Blockly from 'blockly';
  import { registerBlocks, toolbox, buildToolbox } from './blocks.js';
  import ColorPicker from './ColorPicker.svelte';

  registerBlocks();

  let blocklyDiv;
  let workspace;
  let lastMoveToBlock = null;
  let lastMoveByBlock = null;

  // Color picker state
  let selectedPenBlock = null;
  let selectedColorSwatch = null;
  let pickerVisible = false;
  let pickerX = 0;
  let pickerY = 0;
  let pickerSelectedIdx = 0;

  const PRESETS = [
    [255,   0,   0],
    [255, 140,   0],
    [220, 180,   0],
    [  0, 200,   0],
    [  0, 200, 200],
    [  0,  80, 255],
    [160,   0, 240],
  ];

  function closestPresetIdx(r, g, b) {
    let best = 0, bestDist = Infinity;
    for (let i = 0; i < PRESETS.length; i++) {
      const [pr, pg, pb] = PRESETS[i];
      const d = (r-pr)**2 + (g-pg)**2 + (b-pb)**2;
      if (d < bestDist) { bestDist = d; best = i; }
    }
    return best;
  }

  function openPicker(block) {
    if (block.getInput('COLOR_PARAM')?.connection?.targetBlock()) return;
    const r = Number(block.getFieldValue('R'));
    const g = Number(block.getFieldValue('G'));
    const b = Number(block.getFieldValue('B'));
    pickerSelectedIdx = closestPresetIdx(r, g, b);

    const svg = block.getSvgRoot();
    if (svg) {
      const path = svg.querySelector(':scope > .blocklyPath') ?? svg;
      const rect = path.getBoundingClientRect();
      pickerX = rect.left;
      pickerY = rect.bottom + 4;
    }
    pickerVisible = true;
  }

  function openSwatchPicker(field) {
    pickerSelectedIdx = closestPresetIdx(field.r_, field.g_, field.b_);
    const rect = field.rect_?.getBoundingClientRect();
    if (rect) { pickerX = rect.left; pickerY = rect.bottom + 4; }
    selectedColorSwatch = field;
    selectedPenBlock = null;
    pickerVisible = true;
  }

  function onPickerPick(e) {
    const { r, g, b, idx } = e.detail;
    if (selectedPenBlock && !selectedPenBlock.isDisposed()) {
      selectedPenBlock.setFieldValue(String(r), 'R');
      selectedPenBlock.setFieldValue(String(g), 'G');
      selectedPenBlock.setFieldValue(String(b), 'B');
    } else if (selectedColorSwatch && !selectedColorSwatch.getSourceBlock()?.isDisposed()) {
      selectedColorSwatch.setRGB(r, g, b);
    } else {
      pickerVisible = false; return;
    }
    pickerSelectedIdx = idx;
  }

  function onPickerClose() {
    pickerVisible = false;
  }

  class FixedSizeMetricsManager extends Blockly.MetricsManager {
    getContentMetrics(opt_getWorkspaceCoordinates) {
      const scale = opt_getWorkspaceCoordinates ? 1 : this.workspace_.scale;
      const half = 2500;
      return { height: half * 2 * scale, width: half * 2 * scale, top: -half * scale, left: -half * scale };
    }
  }

  class KinderConstantProvider extends Blockly.zelos.ConstantProvider {
    shapeFor(connection) {
      if (connection.getCheck()?.includes('Condition')) return this.HEXAGONAL;
      return super.shapeFor(connection);
    }
  }
  class KinderRenderer extends Blockly.zelos.Renderer {
    makeConstants_() { return new KinderConstantProvider(); }
  }
  Blockly.blockRendering.register('kinder', KinderRenderer);

  const retroTheme = Blockly.Theme.defineTheme('retro', {
    base: Blockly.Themes.Classic,
    blockStyles: {
      param_style: {
        colourPrimary:   '#fef9c3',
        colourSecondary: '#fde047',
        colourTertiary:  '#ca8a04',
      },
      condition_style: {
        colourPrimary:   '#ffedd5',
        colourSecondary: '#fed7aa',
        colourTertiary:  '#ea580c',
      },
    },
    componentStyles: {
      workspaceBackgroundColour: '#080312',
      toolboxBackgroundColour:   '#0f0525',
      toolboxForegroundColour:   '#f3e8ff',
      flyoutBackgroundColour:    '#1a0a3d',
      flyoutForegroundColour:    '#f3e8ff',
      flyoutOpacity:             0.95,
      scrollbarColour:           '#7c3aed',
      scrollbarOpacity:          0.7,
    },
    fontStyle: { family: "'Silkscreen', monospace", size: 12 },
  });

  function setShadowNum(block, inputName, val) {
    const connected = block.getInput(inputName)?.connection?.targetBlock();
    if (connected?.isShadow()) connected.setFieldValue(String(val), 'NUM');
  }

  export function setMoveCoords(x, y) {
    if (!lastMoveToBlock || lastMoveToBlock.isDisposed()) return;
    const snap = v => Math.round(Math.max(-2, Math.min(2, v)) * 10) / 10;
    setShadowNum(lastMoveToBlock, 'INPUT_X', snap(x));
    setShadowNum(lastMoveToBlock, 'INPUT_Y', snap(y));
  }

  export function setMoveDelta(dx, dy) {
    if (!lastMoveByBlock || lastMoveByBlock.isDisposed()) return;
    const snap = v => Math.round(Math.max(-4, Math.min(4, v)) * 10) / 10;
    setShadowNum(lastMoveByBlock, 'INPUT_DX', snap(dx));
    setShadowNum(lastMoveByBlock, 'INPUT_DY', snap(dy));
  }

  let penColorEnabled = true;

  export function setPenColorEnabled(enabled) {
    penColorEnabled = enabled;
    if (workspace) workspace.updateToolbox(buildToolbox(enabled));
    setTimeout(() => updateEnabledStates(), 0);
  }

  const SEQUENCE_HEADS = new Set(['start', 'define_skill']);
  const CUSTOM_COLLAPSE = new Set(['start', 'define_skill', 'use_skill', 'set_pen_color', 'move_base_to_target', 'move_base_by', 'repeat', 'repeat_while', 'pen_up', 'pen_down', 'dip_arm', 'spawn_paint_bucket', 'remove_paint_bucket']);

  function isUseSkillValid(block) {
    const skillName = block.getFieldValue('SKILL');
    if (!skillName || skillName === '__NONE__') return false;
    const skillBlock = workspace?.getAllBlocks(false).find(
      b => b.type === 'define_skill' && b.getFieldValue('NAME') === skillName
    );
    if (!skillBlock) return false;
    const count = parseInt(skillBlock.getFieldValue('ARGS')) || 0;
    const defs = block.paramDefs_ || [];
    if (defs.length !== count) return false;
    if (Object.keys(block.impreciseSaved_ || {}).length > 0) return false;
    if ((block.staleSaved_?.size ?? 0) > 0) return false;
    for (let i = 0; i < (block.paramDefs_?.length ?? 0); i++) {
      const def = block.paramDefs_[i];
      if (def.type !== 'color') {
        const raw = String(block.getFieldValue('ARG_VAL_' + i) ?? '');
        if (raw.toUpperCase() === 'NULL' || isNaN(Number(raw))) return false;
      }
    }
    for (let i = 0; i < count; i++) {
      const expName = skillBlock.getFieldValue('PARAM_NAME_' + i) || ('param' + (i + 1));
      const expType = skillBlock.getFieldValue('PARAM_TYPE_' + i) || 'int';
      if (defs[i].name !== expName || defs[i].type !== expType) return false;
    }
    return true;
  }

  function hasInvalidParamRef(block) {
    for (const input of block.inputList) {
      const connected = input.connection?.targetBlock();
      if (connected?.type === 'param_ref') {
        const val = connected.getFieldValue('PARAM');
        if (!val || val === '__NONE__') return true;
      }
    }
    return false;
  }

  function markReachable(block, reachable) {
    while (block) {
      reachable.add(block.id);
      if (block.type === 'repeat' || block.type === 'repeat_while') markReachable(block.getInputTargetBlock('BODY'), reachable);
      block = block.getNextBlock();
    }
  }

  let _updatingEnabled = false;
  function updateEnabledStates() {
    if (!workspace || _updatingEnabled) return;
    _updatingEnabled = true;
    try {
    const reachable = new Set();
    for (const top of workspace.getTopBlocks(false)) {
      if (SEQUENCE_HEADS.has(top.type)) markReachable(top.getInputTargetBlock('BODY'), reachable);
    }
    for (const block of workspace.getAllBlocks(false)) {
      if (SEQUENCE_HEADS.has(block.type)) continue;
      if (block.isShadow()) continue;
      if (block.type === 'param_ref') continue; // always connectable; dropdown handles invalid context
      if (block.type === 'condition') continue;  // value block; parent repeat_while handles state
      let should = reachable.has(block.id);
      if (should && block.type === 'use_skill') {
        should = isUseSkillValid(block);
      }
      if (should) should = !hasInvalidParamRef(block);
      if (!penColorEnabled && block.type === 'set_pen_color') should = false;
      if (block.isEnabled() !== should) block.setEnabled(should);
    }
    } finally { _updatingEnabled = false; }
  }

  export function selectBlock(blockId) {
    if (!blockId || !workspace) return;
    const block = workspace.getBlockById(blockId);
    if (!block) return;
    Blockly.common.setSelected(block);
  }

  export function hasStartBlock() {
    return workspace?.getAllBlocks(false).some(b => b.type === 'start') ?? false;
  }

  export function hasParamErrors() {
    const skillBodies = {};
    for (const top of workspace.getTopBlocks(false)) {
      if (top.type === 'define_skill') {
        const name = top.getFieldValue('NAME');
        if (name) skillBodies[name] = top.getInputTargetBlock('BODY');
      }
    }
    const visitedSkills = new Set();

    function walkErrors(block) {
      while (block) {
        if (!penColorEnabled && block.type === 'set_pen_color')
          return "This challenge uses paint buckets — remove the Set Pen Color block to run!";
        if (block.type === 'use_skill') {
          const defs = block.paramDefs_ || [];
          for (let i = 0; i < defs.length; i++) {
            if (defs[i].type !== 'color') {
              const raw = String(block.getFieldValue('ARG_VAL_' + i) ?? '');
              if (raw.toUpperCase() === 'NULL')
                return `Skill parameter "${defs[i].name}" has no value — fill it in!`;
              if (isNaN(Number(raw)))
                return `"${raw}" is not a valid number for parameter "${defs[i].name}"!`;
            }
          }
          if (!block.isEnabled())
            return "A skill block has missing or invalid parameters — fix the red blocks first!";
          const skillName = block.getFieldValue('SKILL');
          if (skillName && skillName !== '__NONE__' && !visitedSkills.has(skillName)) {
            visitedSkills.add(skillName);
            const bodyErr = walkErrors(skillBodies[skillName]);
            if (bodyErr) return bodyErr;
          }
        }
        if (block.type === 'repeat' || block.type === 'repeat_while') {
          const bodyErr = walkErrors(block.getInputTargetBlock('BODY'));
          if (bodyErr) return bodyErr;
        }
        block = block.getNextBlock();
      }
      return null;
    }
    for (const top of workspace.getTopBlocks(false)) {
      if (top.type !== 'start') continue;
      const err = walkErrors(top.getInputTargetBlock('BODY'));
      if (err) return err;
    }
    return null;
  }

  export function getProgram() {
    const blocks = [];

    // Map skill name → first body block of its define_skill
    const skillBodies = {};
    for (const top of workspace.getTopBlocks(false)) {
      if (top.type === 'define_skill') {
        const name = top.getFieldValue('NAME');
        if (name) skillBodies[name] = top.getInputTargetBlock('BODY');
      }
    }

    function getNumFromInput(block, inputName, params) {
      const connected = block.getInput(inputName)?.connection?.targetBlock();
      if (!connected) return 0;
      if (connected.type === 'param_ref') {
        const scale = Number(connected.getFieldValue('SCALE') ?? 1);
        const val = params?.[connected.getFieldValue('PARAM')];
        return (val != null && !isNaN(Number(val))) ? scale * Number(val) : 0;
      }
      return Number(connected.getFieldValue('NUM')) || 0;
    }

    // Inline-expand a chain of blocks into `output`, substituting skill calls recursively.
    // repeat_while is sent as a nested structure; repeat is pre-expanded inline.
    function expandChain(block, params, depth, output) {
      if (depth > 20) return;
      while (block) {
        if (!block.isEnabled()) { block = block.getNextBlock(); continue; }

        if (block.type === 'repeat') {
          const count = Math.max(0, Math.min(100, Math.round(getNumFromInput(block, 'INPUT_COUNT', params))));
          const body = block.getInputTargetBlock('BODY');
          if (body) for (let i = 0; i < count; i++) expandChain(body, params, depth + 1, output);
        } else if (block.type === 'repeat_while') {
          const condBlock = block.getInput('CONDITION')?.connection?.targetBlock();
          if (condBlock?.type === 'condition') {
            const varField = condBlock.getFieldValue('VAR') || 'X';
            let serializedVar;
            if (varField === 'X' || varField === 'Y') {
              serializedVar = varField;
            } else {
              const paramVal = params?.[varField];
              serializedVar = (paramVal != null && !isNaN(Number(paramVal))) ? String(paramVal) : '0';
            }
            const entry = {
              type: 'repeat_while',
              blockId: block.id,
              var:  serializedVar,
              op:   condBlock.getFieldValue('OP')  || '>',
              threshold: getNumFromInput(condBlock, 'THRESHOLD', params),
              body: [],
            };
            expandChain(block.getInputTargetBlock('BODY'), params, depth + 1, entry.body);
            output.push(entry);
          }
        } else if (block.type === 'use_skill') {
          const skillName = block.getFieldValue('SKILL');
          const bodyStart = skillBodies[skillName];
          if (bodyStart) {
            const defs = block.paramDefs_ || [];
            const callParams = {};
            for (let i = 0; i < defs.length; i++) {
              const { name, type } = defs[i];
              if (type === 'color') {
                const f = block.getField('ARG_COLOR_' + i);
                callParams[name] = f ? { type: 'color', r: f.r_, g: f.g_, b: f.b_ } : null;
              } else {
                const raw = block.getFieldValue('ARG_VAL_' + i);
                callParams[name] = (raw == null || String(raw).toUpperCase() === 'NULL') ? null : Number(raw);
              }
            }
            expandChain(bodyStart, callParams, depth + 1, output);
          }
        } else {
          const entry = { type: block.type, blockId: block.id };
          if (block.type === 'move_base_to_target') {
            entry.x = getNumFromInput(block, 'INPUT_X', params);
            entry.y = getNumFromInput(block, 'INPUT_Y', params);
          } else if (block.type === 'move_base_by') {
            entry.dx = getNumFromInput(block, 'INPUT_DX', params);
            entry.dy = getNumFromInput(block, 'INPUT_DY', params);
          } else if (block.type === 'set_pen_color') {
            const colorBlock = block.getInput('COLOR_PARAM')?.connection?.targetBlock();
            if (colorBlock?.type === 'param_ref') {
              const colorVal = params?.[colorBlock.getFieldValue('PARAM')];
              entry.r = colorVal?.type === 'color' ? colorVal.r : 0;
              entry.g = colorVal?.type === 'color' ? colorVal.g : 0;
              entry.b = colorVal?.type === 'color' ? colorVal.b : 0;
            } else {
              entry.r = Number(block.getFieldValue('R'));
              entry.g = Number(block.getFieldValue('G'));
              entry.b = Number(block.getFieldValue('B'));
            }
          } else if (block.type === 'spawn_paint_bucket') {
            const f = block.getField('COLOR');
            entry.x = getNumFromInput(block, 'INPUT_X', params);
            entry.y = getNumFromInput(block, 'INPUT_Y', params);
            entry.r = f?.r_ ?? 255;
            entry.g = f?.g_ ?? 0;
            entry.b = f?.b_ ?? 0;
          }
          output.push(entry);
        }
        block = block.getNextBlock();
      }
    }

    for (const top of workspace.getTopBlocks(true)) {
      if (top.type !== 'start') continue;
      expandChain(top.getInputTargetBlock('BODY'), {}, 0, blocks);
      break;
    }
    return { blocks };
  }

  function onWheel(e) {
    e.preventDefault();
    if (e.ctrlKey) {
      const rect = blocklyDiv.getBoundingClientRect();
      workspace.zoom(e.clientX - rect.left, e.clientY - rect.top, e.deltaY < 0 ? 1 : -1);
    } else {
      const dx = e.deltaX + (e.shiftKey ? e.deltaY : 0);
      const dy = e.shiftKey ? 0 : e.deltaY;
      workspace.scroll(workspace.scrollX - dx, workspace.scrollY - dy);
    }
  }

  function onDblClick(e) {
    if (e.target.closest('.blocklyDraggable')) return; // ignore double-clicks on blocks
    workspace.setScale(1.5);
    requestAnimationFrame(() => {
      workspace.scroll(blocklyDiv.clientWidth / 2, blocklyDiv.clientHeight / 2);
    });
  }

  function onKeyDown(e) {
    if (!e.ctrlKey) return;
    if (e.key === '=' || e.key === '+') { e.preventDefault(); workspace.zoomCenter(1); }
    else if (e.key === '-')             { e.preventDefault(); workspace.zoomCenter(-1); }
  }

  let pendingDeleteStartId = null;
  let saveTimer = null;
  const WS_KEY = 'kinder-blockly-ws-v3';
  // Set after the first visit so we never show the starter program again,
  // even if the user clears their workspace later.
  const VISITED_KEY = 'kinder-blockly-visited';

  // Starter workspace shown only on a browser's very first visit: pen down,
  // move left by 1, move up by 1. Gives new students something to read and
  // a Run button that produces a visible result immediately.
  const STARTER_WORKSPACE = {
    blocks: {
      languageVersion: 0,
      blocks: [{
        type: 'start',
        // Placed up-and-left of world origin so the block sits roughly in the
        // upper-left of the visible workspace after the default scroll(divW/2,
        // divH/2). Without the negative offset the block lands in the bottom-
        // right of the viewport and clips off the right edge.
        x: -180,
        y: -120,
        inputs: {
          BODY: {
            block: {
              type: 'pen_down',
              next: {
                block: {
                  type: 'move_base_by',
                  inputs: {
                    INPUT_DX: { shadow: { type: 'kinder_num', fields: { NUM: -1 } } },
                    INPUT_DY: { shadow: { type: 'kinder_num', fields: { NUM: 0 } } },
                  },
                  next: {
                    block: {
                      type: 'move_base_by',
                      inputs: {
                        INPUT_DX: { shadow: { type: 'kinder_num', fields: { NUM: 0 } } },
                        INPUT_DY: { shadow: { type: 'kinder_num', fields: { NUM: 1 } } },
                      },
                    },
                  },
                },
              },
            },
          },
        },
      }],
    },
  };

  function saveWorkspace() {
    if (!workspace) return;
    try {
      const next = JSON.stringify(Blockly.serialization.workspaces.save(workspace));
      const prev = localStorage.getItem(WS_KEY);
      if (prev) localStorage.setItem(WS_KEY + '-backup', prev);
      localStorage.setItem(WS_KEY, next);
    } catch {}
  }

  function onPointerUp() {
    if (!pendingDeleteStartId) return;
    const block = workspace?.getBlockById(pendingDeleteStartId);
    if (block && !block.isDisposed()) block.dispose(false);
    pendingDeleteStartId = null;
  }

  onMount(async () => {
    // Wait for Silkscreen before injecting Blockly. Blockly measures glyph
    // widths during block construction, and if the retro font has not yet
    // arrived blocks get sized with fallback-font metrics and look wrong
    // until the next reload. document.fonts.ready is not enough here: with
    // Google Fonts + display=swap the woff2 fetch can be deferred past the
    // point where `ready` resolves, so request the specific face explicitly.
    try { await document.fonts.load("12px 'Silkscreen'"); } catch {}

    // Connection snap radii. Blockly's defaults (28 / 28 in workspace coords)
    // are tuned for small zoom-1.0 blocks; at our 1.5× scale with chunky
    // Silkscreen glyphs, students see two shapes visually overlapping but
    // the underlying connection points are still further than 28 wsu apart,
    // so the snap never fires. Bumping both radii lets the block snap as
    // soon as the input clearly overlaps the opening.
    Blockly.config.snapRadius = 60;
    Blockly.config.connectingSnapRadius = 80;

    workspace = Blockly.inject(blocklyDiv, {
      toolbox,
      theme: retroTheme,
      trashcan: true,
      scrollbars: true,
      renderer: 'kinder',
      grid: { spacing: 25, length: 3, colour: '#5b21b6', snap: true },
      zoom: { controls: false, wheel: false, startScale: 1.5, maxScale: 4, minScale: 0.2, scaleSpeed: 1.2 },
      plugins: { metricsManager: FixedSizeMetricsManager },
    });

    for (const id of ['blockDisable', 'blockCollapse', 'blockCollapseExpand',
                       'collapseWorkspace', 'expandWorkspace', 'cleanWorkspace']) {
      try { Blockly.ContextMenuRegistry.registry.unregister(id); } catch (_) {}
    }

    // Blockly's default DELETE_X_BLOCKS label is "Delete %1 Blocks", where %1
    // is computed from the block + every descendant in connection inputs.
    // For our compound blocks (start/define_skill bodies, shadow number
    // children inside movement inputs) that count is misleading — students
    // see "Delete 4 Blocks" when right-clicking a single visible block.
    // Drop the count entirely.
    Blockly.Msg.DELETE_X_BLOCKS = 'Delete Blocks';

    try { Blockly.ContextMenuRegistry.registry.unregister('duplicateFromHere'); } catch (_) {}
    Blockly.ContextMenuRegistry.registry.register({
      id: 'duplicateFromHere',
      weight: 1,
      scopeType: Blockly.ContextMenuRegistry.ScopeType.BLOCK,
      displayText: 'Duplicate from here',
      preconditionFn: (scope) => scope.block.type !== 'start' ? 'enabled' : 'hidden',
      callback: (scope) => {
        const block = scope.block;
        const ws = block.workspace;
        const state = Blockly.serialization.blocks.save(block, { addCoordinates: false });
        const xy = block.getRelativeToSurfaceXY();
        state.x = xy.x + 40;
        state.y = xy.y + 40;
        const newBlock = Blockly.serialization.blocks.append(state, ws);
        function collapseChain(b) {
          while (b) {
            if (CUSTOM_COLLAPSE.has(b.type)) b.customCollapse_?.();
            else if (!b.isCollapsed() && !b.outputConnection) b.setCollapsed(true);
            if (b.type === 'repeat' || b.type === 'repeat_while') collapseChain(b.getInputTargetBlock('BODY'));
            b = b.getNextBlock();
          }
        }
        collapseChain(newBlock);
      },
    });

    const s = document.createElement('style');
    s.id = 'blockly-silkscreen';
    s.textContent =
      '.blocklyTreeLabel{font-family:"Silkscreen",monospace!important;font-size:22px!important}' +
      '.blocklyHtmlInput{font-family:"Silkscreen",monospace!important;font-size:20px!important}' +
      '.blocklyTreeRow{padding-top:14px!important;padding-bottom:14px!important;height:auto!important}' +
      '.blocklyContextMenu{font-size:20px!important}' +
      '.blocklyMenuItem{padding:10px 24px!important;min-height:unset!important}' +
      '.blocklyMenuItemContent{font-size:20px!important}' +
      '.kinder-param-dropdown .blocklyMenuItemContent{color:#1a1a1a!important}' +
      '.kinder-param-dropdown .blocklyMenuItem{color:#1a1a1a!important}';
    document.head.appendChild(s);

    workspace.addChangeListener(e => {
      if (e.type === 'selected') {
        // Collapse the block that just lost focus; expand the one gaining it.
        // Exception: if a value block (output) is being picked up to drop into an input,
        // keep the old block expanded so its value input slots remain accessible.
        const block = e.newElementId ? workspace.getBlockById(e.newElementId) : null;
        if (!block?.outputConnection) {
          if (e.oldElementId) {
            const old = workspace.getBlockById(e.oldElementId);
            if (CUSTOM_COLLAPSE.has(old?.type)) old.customCollapse_?.();
            else if (old && !old.isCollapsed() && !old.outputConnection) old.setCollapsed(true);
          }
        }
        if (CUSTOM_COLLAPSE.has(block?.type)) block.customExpand_?.();
        else if (block?.isCollapsed() && !block.outputConnection) block.setCollapsed(false);

        if (block?.type === 'move_base_to_target') { lastMoveToBlock = block; lastMoveByBlock = null; }
        else if (block?.type === 'move_base_by')   { lastMoveByBlock = block; lastMoveToBlock = null; }
        else if (block)                             { lastMoveToBlock = null;  lastMoveByBlock = null; }
        // Deselection keeps both so canvas interaction still works

        // Track selected pen block; close picker if selection changes away from it
        if (block?.type === 'set_pen_color') {
          selectedPenBlock = block;
          selectedColorSwatch = null;
        } else {
          selectedPenBlock = null;
          if (block?.type !== 'use_skill') { selectedColorSwatch = null; pickerVisible = false; }
        }
      }
    });

    workspace.addChangeListener(evt => {
      if (evt.type === Blockly.Events.BLOCK_CREATE) {
        const block = workspace.getBlockById(evt.blockId);
        if (block?.type === 'use_skill') {
          const blockId = evt.blockId;
          setTimeout(() => {
            const b = workspace?.getBlockById(blockId);
            if (b && !b.isDisposed()) { b.updateParamInputs_?.(); updateEnabledStates(); }
          }, 0);
        }
        if (CUSTOM_COLLAPSE.has(block?.type)) block.customCollapse_?.();
        else if (block && !block.isCollapsed() && !block.outputConnection) block.setCollapsed(true);
        if (block?.type === 'start') {
          const startBlocks = workspace.getAllBlocks(false).filter(b => b.type === 'start');
          if (startBlocks.length > 1) {
            pendingDeleteStartId = block.id;
            dispatch('message', "You can only start once!");
          }
        }
      }
      if ([Blockly.Events.BLOCK_MOVE, Blockly.Events.BLOCK_CREATE, Blockly.Events.BLOCK_DELETE].includes(evt.type)) {
        updateEnabledStates();
      }
      if (evt.type === Blockly.Events.BLOCK_CHANGE && evt.element === 'field') {
        const changed = workspace.getBlockById(evt.blockId);
        if (changed?.type === 'define_skill') {
          for (const b of workspace.getAllBlocks(false)) {
            if (b.type === 'use_skill') b.updateParamInputs_?.();
          }
          updateEnabledStates();
        } else if (changed?.type === 'param_ref') {
          updateEnabledStates();
        } else if (changed?.type === 'use_skill') {
          updateEnabledStates();
        }
      }
    });

    // Color picker click handling (set_pen_color re-click + FieldColorSwatch click)
    blocklyDiv.addEventListener('click', e => {
      if (e.target?.tagName?.toLowerCase() === 'input') return;

      // FieldColorSwatch clicked
      if (e.target?.getAttribute?.('data-cs')) {
        const field = e.target._csField;
        if (field && !field.getSourceBlock()?.isDisposed()) {
          if (pickerVisible && selectedColorSwatch === field) pickerVisible = false;
          else openSwatchPicker(field);
          return;
        }
      }

      // set_pen_color re-click
      if (!selectedPenBlock || selectedPenBlock.isDisposed()) {
        if (pickerVisible) { pickerVisible = false; selectedColorSwatch = null; }
        return;
      }
      const svg = selectedPenBlock.getSvgRoot();
      if (svg && svg.contains(e.target)) {
        pickerVisible ? (pickerVisible = false) : openPicker(selectedPenBlock);
      } else if (pickerVisible) {
        pickerVisible = false;
      }
    });

    // Persist workspace across hot reloads (v2: movement blocks use value inputs)

    workspace.addChangeListener(() => {
      clearTimeout(saveTimer);
      saveTimer = setTimeout(saveWorkspace, 400);
    });

    // Recalculate flyout position after CSS and layout have settled, then centre on origin
    requestAnimationFrame(() => {
      Blockly.svgResize(workspace);
      const saved = localStorage.getItem(WS_KEY);
      if (saved) {
        try {
          Blockly.serialization.workspaces.load(JSON.parse(saved), workspace);
          // Refresh use_skill blocks now that all define_skill blocks are loaded.
          // (During deserialization the SKILL field change fires before define_skill
          // blocks exist, so updateParamInputs_ may have had nothing to look up.)
          for (const block of workspace.getAllBlocks(false)) {
            if (block.type === 'use_skill') block.updateParamInputs_?.();
          }
          for (const block of workspace.getAllBlocks(false)) {
            if (block.isShadow() || block.outputConnection) continue;
            if (CUSTOM_COLLAPSE.has(block.type)) block.customCollapse_?.();
            else if (!block.isCollapsed()) block.setCollapsed(true);
          }
        } catch { localStorage.removeItem(WS_KEY); }
      }
      // One-time migration from v2 (next connections) → v3 (BODY statement input).
      if (!saved) {
        const v2 = localStorage.getItem('kinder-blockly-ws-v2');
        if (v2) {
          try {
            const ws = JSON.parse(v2);
            function migrateBlock(blk) {
              if (!blk) return;
              for (const inp of Object.values(blk.inputs || {})) {
                if (inp.block) migrateBlock(inp.block);
                if (inp.shadow) migrateBlock(inp.shadow);
              }
              if (blk.next?.block) migrateBlock(blk.next.block);
              if ((blk.type === 'start' || blk.type === 'define_skill') && blk.next) {
                blk.inputs = blk.inputs || {};
                blk.inputs['BODY'] = blk.next;
                delete blk.next;
              }
            }
            for (const blk of ws.blocks?.blocks || []) migrateBlock(blk);
            localStorage.setItem(WS_KEY, JSON.stringify(ws));
            Blockly.serialization.workspaces.load(ws, workspace);
            for (const block of workspace.getAllBlocks(false)) {
              if (block.type === 'use_skill') block.updateParamInputs_?.();
            }
            for (const block of workspace.getAllBlocks(false)) {
              if (block.isShadow() || block.outputConnection) continue;
              if (CUSTOM_COLLAPSE.has(block.type)) block.customCollapse_?.();
              else if (!block.isCollapsed()) block.setCollapsed(true);
            }
          } catch {}
        }
      }

      // First-visit starter program: only loads if no prior workspace exists in
      // either v3 or v2 storage AND this browser has never been here before.
      // Subsequent visits skip this, even if the user has cleared their work.
      if (workspace.getTopBlocks(false).length === 0 && !localStorage.getItem(VISITED_KEY)) {
        try {
          Blockly.serialization.workspaces.load(STARTER_WORKSPACE, workspace);
          for (const block of workspace.getAllBlocks(false)) {
            if (block.isShadow() || block.outputConnection) continue;
            if (CUSTOM_COLLAPSE.has(block.type)) block.customCollapse_?.();
            else if (!block.isCollapsed()) block.setCollapsed(true);
          }
        } catch {}
      }
      localStorage.setItem(VISITED_KEY, '1');

      workspace.setScale(1.5);
      workspace.scroll(blocklyDiv.clientWidth / 2, blocklyDiv.clientHeight / 2);
      updateEnabledStates();

      // Belt-and-braces guard: even with the pre-inject await, the FontFace
      // for Silkscreen could in principle still be settling (e.g. cached
      // bytes that have not finished decoding) by the time the first blocks
      // render. Re-render once the specific face is fully ready. We use
      // document.fonts.load rather than document.fonts.ready because the
      // latter resolves whenever nothing is currently loading, which is not
      // the same as "Silkscreen is available".
      document.fonts?.load("12px 'Silkscreen'").then(() => {
        if (!workspace || workspace.isDisposed?.()) return;
        for (const block of workspace.getAllBlocks(false)) {
          block.render(false);
        }
        Blockly.svgResize(workspace);
      }).catch(() => {});
    });

    blocklyDiv.addEventListener('wheel', onWheel, { passive: false });
    blocklyDiv.addEventListener('dblclick', onDblClick);
    document.addEventListener('pointerup', onPointerUp);
    document.addEventListener('keydown', onKeyDown);
  });

  onDestroy(() => {
    clearTimeout(saveTimer);
    if (workspace) saveWorkspace();
    blocklyDiv?.removeEventListener('wheel', onWheel);
    blocklyDiv?.removeEventListener('dblclick', onDblClick);
    document.removeEventListener('pointerup', onPointerUp);
    document.removeEventListener('keydown', onKeyDown);
    workspace?.dispose();
    clearTimeout(saveTimer); // kill any timer re-armed by dispose's change events
    document.getElementById('blockly-silkscreen')?.remove();
    try { Blockly.ContextMenuRegistry.registry.unregister('duplicateFromHere'); } catch (_) {}
  });
</script>

<div id="blockly-area">
  <div id="blockly-div" bind:this={blocklyDiv}></div>
</div>

{#if pickerVisible}
  <ColorPicker
    x={pickerX}
    y={pickerY}
    selectedIdx={pickerSelectedIdx}
    on:pick={onPickerPick}
    on:close={onPickerClose}
  />
{/if}

<style>
  #blockly-area { flex: 2; position: relative; background: var(--bg); }
  #blockly-div  { position: absolute; inset: 0; }
</style>
