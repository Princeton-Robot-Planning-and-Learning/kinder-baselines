<script>
  import { onMount, onDestroy } from 'svelte';
  import * as Blockly from 'blockly';
  import { registerBlocks, toolbox } from './blocks.js';

  registerBlocks();

  let blocklyDiv;
  let workspace;

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

  export function getProgram() {
    const blocks = [];
    for (const top of workspace.getTopBlocks(true)) {
      let block = top;
      while (block) {
        const entry = { type: block.type };
        if (block.type === 'move_base_to_target') {
          entry.x = block.getFieldValue('X');
          entry.y = block.getFieldValue('Y');
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
      zoom: { controls: false, wheel: false, startScale: 1.0, maxScale: 4, minScale: 0.2, scaleSpeed: 1.2 },
      plugins: { metricsManager: FixedSizeMetricsManager },
    });

    const s = document.createElement('style');
    s.id = 'blockly-silkscreen';
    s.textContent =
      '.blocklyTreeLabel{font-family:"Silkscreen",monospace!important;font-size:22px!important}' +
      '.blocklyHtmlInput{font-family:"Silkscreen",monospace!important;font-size:20px!important}' +
      '.blocklyTreeRow{padding-top:14px!important;padding-bottom:14px!important;height:auto!important}';
    document.head.appendChild(s);

    // Recalculate flyout position after CSS and layout have settled
    requestAnimationFrame(() => Blockly.svgResize(workspace));

    blocklyDiv.addEventListener('wheel', onWheel, { passive: false });
    document.addEventListener('keydown', onKeyDown);
  });

  onDestroy(() => {
    blocklyDiv?.removeEventListener('wheel', onWheel);
    document.removeEventListener('keydown', onKeyDown);
    workspace?.dispose();
    document.getElementById('blockly-silkscreen')?.remove();
  });
</script>

<div id="blockly-area">
  <div id="blockly-div" bind:this={blocklyDiv}></div>
</div>

<style>
  #blockly-area { flex: 2; position: relative; background: var(--bg); }
  #blockly-div  { position: absolute; inset: 0; }
</style>
