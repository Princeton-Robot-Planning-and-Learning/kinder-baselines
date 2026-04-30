<script>
  import { onMount, onDestroy } from 'svelte';
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
    const r = Number(block.getFieldValue('R'));
    const g = Number(block.getFieldValue('G'));
    const b = Number(block.getFieldValue('B'));
    pickerSelectedIdx = closestPresetIdx(r, g, b);

    const svg = block.getSvgRoot();
    if (svg) {
      const rect = svg.getBoundingClientRect();
      pickerX = rect.left;
      pickerY = rect.bottom + 4;
    }
    pickerVisible = true;
  }

  function onPickerPick(e) {
    if (!selectedPenBlock || selectedPenBlock.isDisposed()) { pickerVisible = false; return; }
    const { r, g, b, idx } = e.detail;
    selectedPenBlock.setFieldValue(String(r), 'R');
    selectedPenBlock.setFieldValue(String(g), 'G');
    selectedPenBlock.setFieldValue(String(b), 'B');
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

  export function setMoveCoords(x, y) {
    if (!lastMoveToBlock || lastMoveToBlock.isDisposed()) return;
    const snap = v => Math.round(Math.max(-2, Math.min(2, v)) * 10) / 10;
    lastMoveToBlock.setFieldValue(snap(x), 'X');
    lastMoveToBlock.setFieldValue(snap(y), 'Y');
  }

  export function setMoveDelta(dx, dy) {
    if (!lastMoveByBlock || lastMoveByBlock.isDisposed()) return;
    const snap = v => Math.round(Math.max(-4, Math.min(4, v)) * 10) / 10;
    lastMoveByBlock.setFieldValue(snap(dx), 'DX');
    lastMoveByBlock.setFieldValue(snap(dy), 'DY');
  }

  function updateEnabledStates() {
    if (!workspace) return;
    const reachable = new Set();
    for (const top of workspace.getTopBlocks(false)) {
      if (top.type === 'start') {
        let b = top.getNextBlock();
        while (b) { reachable.add(b.id); b = b.getNextBlock(); }
      }
    }
    for (const block of workspace.getAllBlocks(false)) {
      if (block.type === 'start') continue;
      const should = reachable.has(block.id);
      if (block.isEnabled() !== should) block.setEnabled(should);
    }
  }

  export function getProgram() {
    const blocks = [];
    for (const top of workspace.getTopBlocks(true)) {
      let block = top;
      while (block) {
        if (block.type === 'start') { block = block.getNextBlock(); continue; }
        if (!block.isEnabled()) { block = block.getNextBlock(); continue; }
        const entry = { type: block.type };
        if (block.type === 'move_base_to_target') {
          entry.x = block.getFieldValue('X');
          entry.y = block.getFieldValue('Y');
        } else if (block.type === 'move_base_by') {
          entry.dx = block.getFieldValue('DX');
          entry.dy = block.getFieldValue('DY');
        } else if (block.type === 'set_pen_color') {
          entry.r = Number(block.getFieldValue('R'));
          entry.g = Number(block.getFieldValue('G'));
          entry.b = Number(block.getFieldValue('B'));
        }
        blocks.push(entry);
        block = block.getNextBlock();
      }
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

    const s = document.createElement('style');
    s.id = 'blockly-silkscreen';
    s.textContent =
      '.blocklyTreeLabel{font-family:"Silkscreen",monospace!important;font-size:22px!important}' +
      '.blocklyHtmlInput{font-family:"Silkscreen",monospace!important;font-size:20px!important}' +
      '.blocklyTreeRow{padding-top:14px!important;padding-bottom:14px!important;height:auto!important}' +
      '.blocklyContextMenu{font-size:20px!important}' +
      '.blocklyMenuItem{padding:10px 24px!important;min-height:unset!important}' +
      '.blocklyMenuItemContent{font-size:20px!important}';
    document.head.appendChild(s);

    workspace.addChangeListener(e => {
      if (e.type === 'selected') {
        const block = e.newElementId ? workspace.getBlockById(e.newElementId) : null;
        if (block?.type === 'move_base_to_target') { lastMoveToBlock = block; lastMoveByBlock = null; }
        else if (block?.type === 'move_base_by')   { lastMoveByBlock = block; lastMoveToBlock = null; }
        else if (block)                             { lastMoveToBlock = null;  lastMoveByBlock = null; }
        // Deselection keeps both so canvas interaction still works

        // Track selected pen block; close picker if selection changes away from it
        if (block?.type === 'set_pen_color') {
          selectedPenBlock = block;
        } else {
          selectedPenBlock = null;
          pickerVisible = false;
        }
      }
    });

    workspace.addChangeListener(evt => {
      if ([Blockly.Events.BLOCK_MOVE, Blockly.Events.BLOCK_CREATE, Blockly.Events.BLOCK_DELETE].includes(evt.type)) {
        updateEnabledStates();
      }
    });

    // Detect re-click on an already-selected set_pen_color block to open picker.
    // Using 'click' (not 'pointerdown') so drags don't trigger it.
    blocklyDiv.addEventListener('click', e => {
      if (e.target?.tagName?.toLowerCase() === 'input') return; // ignore field edits
      if (!selectedPenBlock || selectedPenBlock.isDisposed()) { pickerVisible = false; return; }
      const svg = selectedPenBlock.getSvgRoot();
      if (svg && svg.contains(e.target)) {
        pickerVisible ? (pickerVisible = false) : openPicker(selectedPenBlock);
      } else if (pickerVisible) {
        pickerVisible = false;
      }
    });

    // Recalculate flyout position after CSS and layout have settled, then centre on origin
    requestAnimationFrame(() => {
      Blockly.svgResize(workspace);
      workspace.scroll(blocklyDiv.clientWidth / 2, blocklyDiv.clientHeight / 2);
      updateEnabledStates();
    });

    blocklyDiv.addEventListener('wheel', onWheel, { passive: false });
    blocklyDiv.addEventListener('dblclick', onDblClick);
    document.addEventListener('keydown', onKeyDown);
  });

  onDestroy(() => {
    blocklyDiv?.removeEventListener('wheel', onWheel);
    blocklyDiv?.removeEventListener('dblclick', onDblClick);
    document.removeEventListener('keydown', onKeyDown);
    workspace?.dispose();
    document.getElementById('blockly-silkscreen')?.remove();
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
