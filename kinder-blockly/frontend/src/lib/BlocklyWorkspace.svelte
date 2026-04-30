<script>
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher();
  import * as Blockly from 'blockly';
  import { registerBlocks, toolbox } from './blocks.js';
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

  const retroTheme = Blockly.Theme.defineTheme('retro', {
    base: Blockly.Themes.Classic,
    blockStyles: {
      param_style: {
        colourPrimary:   '#fef9c3',
        colourSecondary: '#fde047',
        colourTertiary:  '#ca8a04',
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

  const SEQUENCE_HEADS = new Set(['start', 'define_skill']);
  const CUSTOM_COLLAPSE = new Set(['define_skill', 'use_skill', 'set_pen_color', 'move_base_to_target', 'move_base_by']);

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
      if (def.type !== 'color' && String(block.getFieldValue('ARG_VAL_' + i) ?? '').toUpperCase() === 'NULL') return false;
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

  function updateEnabledStates() {
    if (!workspace) return;
    const reachable = new Set();
    for (const top of workspace.getTopBlocks(false)) {
      if (SEQUENCE_HEADS.has(top.type)) {
        let b = top.getNextBlock();
        while (b) { reachable.add(b.id); b = b.getNextBlock(); }
      }
    }
    for (const block of workspace.getAllBlocks(false)) {
      if (SEQUENCE_HEADS.has(block.type)) continue;
      if (block.isShadow()) continue;
      if (block.type === 'param_ref') continue; // always connectable; dropdown handles invalid context
      let should = reachable.has(block.id);
      if (should && block.type === 'use_skill') should = isUseSkillValid(block);
      if (should) should = !hasInvalidParamRef(block);
      if (block.isEnabled() !== should) block.setEnabled(should);
    }
  }

  export function hasParamErrors() {
    for (const top of workspace.getTopBlocks(false)) {
      if (top.type !== 'start') continue;
      let block = top.getNextBlock();
      while (block) {
        if (block.type === 'use_skill' && !block.isEnabled()) return true;
        block = block.getNextBlock();
      }
    }
    return false;
  }

  export function getProgram() {
    const blocks = [];

    // Map skill name → first body block of its define_skill
    const skillBodies = {};
    for (const top of workspace.getTopBlocks(false)) {
      if (top.type === 'define_skill') {
        const name = top.getFieldValue('NAME');
        if (name) skillBodies[name] = top.getNextBlock();
      }
    }

    function getNumFromInput(block, inputName, params) {
      const connected = block.getInput(inputName)?.connection?.targetBlock();
      if (!connected) return 0;
      if (connected.type === 'param_ref') {
        const val = params?.[connected.getFieldValue('PARAM')];
        return (val != null && !isNaN(Number(val))) ? Number(val) : 0;
      }
      return Number(connected.getFieldValue('NUM') ?? 0);
    }

    // Inline-expand a chain of blocks, substituting skill calls recursively.
    // `params` maps param name → value for the current call frame.
    function expandChain(block, params, depth) {
      if (depth > 20) return; // guard against infinite recursion
      while (block) {
        if (!block.isEnabled()) { block = block.getNextBlock(); continue; }

        if (block.type === 'use_skill') {
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
            expandChain(bodyStart, callParams, depth + 1);
          }
        } else {
          const entry = { type: block.type };
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
          }
          blocks.push(entry);
        }
        block = block.getNextBlock();
      }
    }

    for (const top of workspace.getTopBlocks(true)) {
      if (top.type !== 'start') continue;
      expandChain(top.getNextBlock(), {}, 0);
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

  function onPointerUp() {
    if (!pendingDeleteStartId) return;
    const block = workspace?.getBlockById(pendingDeleteStartId);
    if (block && !block.isDisposed()) block.dispose(false);
    pendingDeleteStartId = null;
  }

  onMount(() => {
    workspace = Blockly.inject(blocklyDiv, {
      toolbox,
      theme: retroTheme,
      trashcan: true,
      scrollbars: true,
      renderer: 'zelos',
      grid: { spacing: 25, length: 3, colour: '#5b21b6', snap: true },
      zoom: { controls: false, wheel: false, startScale: 1.5, maxScale: 4, minScale: 0.2, scaleSpeed: 1.2 },
      plugins: { metricsManager: FixedSizeMetricsManager },
    });

    try { Blockly.ContextMenuRegistry.registry.unregister('blockDisable'); } catch (_) {}
    try { Blockly.ContextMenuRegistry.registry.unregister('blockCollapse'); } catch (_) {}
    try { Blockly.ContextMenuRegistry.registry.unregister('blockCollapseExpand'); } catch (_) {}

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
        let b = newBlock;
        while (b) { if (!b.isCollapsed()) b.setCollapsed(true); b = b.getNextBlock(); }
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
    const WS_KEY = 'kinder-blockly-ws-v2';
    workspace.addChangeListener(() => {
      clearTimeout(saveTimer);
      saveTimer = setTimeout(() => {
        try { localStorage.setItem(WS_KEY, JSON.stringify(Blockly.serialization.workspaces.save(workspace))); } catch {}
      }, 400);
    });

    // Recalculate flyout position after CSS and layout have settled, then centre on origin
    requestAnimationFrame(() => {
      Blockly.svgResize(workspace);
      const saved = localStorage.getItem(WS_KEY);
      if (saved) {
        try { Blockly.serialization.workspaces.load(JSON.parse(saved), workspace); } catch { localStorage.removeItem(WS_KEY); }
      }
      workspace.setScale(1.5);
      workspace.scroll(blocklyDiv.clientWidth / 2, blocklyDiv.clientHeight / 2);
      updateEnabledStates();
    });

    blocklyDiv.addEventListener('wheel', onWheel, { passive: false });
    blocklyDiv.addEventListener('dblclick', onDblClick);
    document.addEventListener('pointerup', onPointerUp);
    document.addEventListener('keydown', onKeyDown);
  });

  onDestroy(() => {
    blocklyDiv?.removeEventListener('wheel', onWheel);
    blocklyDiv?.removeEventListener('dblclick', onDblClick);
    document.removeEventListener('pointerup', onPointerUp);
    document.removeEventListener('keydown', onKeyDown);
    workspace?.dispose();
    document.getElementById('blockly-silkscreen')?.remove();
    try { Blockly.ContextMenuRegistry.registry.unregister('duplicateFromHere'); } catch (_) {}
    clearTimeout(saveTimer);
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
