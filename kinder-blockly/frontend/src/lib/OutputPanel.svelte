<script>
  import { onMount, createEventDispatcher } from 'svelte';
  import { drawCanvasTrail, drawPenMarkers, canvasToWorld, drawHoverMarker, drawVectorDrag } from './canvas.js';

  const dispatch = createEventDispatcher();

  export let frameDataUrl = '';
  export let frameInfo = 'DRAG BLOCKS AND CLICK RUN';
  export let studentTrail = [];
  export let targetTrail = [];
  export let score = null;
  export let studentPenEvents = [];
  export let panelWidth = 0;
  export let canGoPrev = false;
  export let canGoNext = false;

  let el;
  let frameImg;
  let targetCanvas;
  let hoverCanvas;
  let trailCanvas;
  let targetLabel;
  let trailLabel;
  let canvasSize = 220;
  let targetLabelPad = 0;

  let dragOriginCanvas = null;
  let dragOriginWorld  = null;
  let isDragging       = false;

  function canvasPos(e) {
    const r  = hoverCanvas.getBoundingClientRect();
    const sx = hoverCanvas.width  / r.width;
    const sy = hoverCanvas.height / r.height;
    return [(e.clientX - r.left) * sx, (e.clientY - r.top) * sy];
  }

  function getSnapped(e) {
    const [px, py] = canvasPos(e);
    const [wx, wy] = canvasToWorld(px, py, hoverCanvas.width, hoverCanvas.height);
    const snap = v => Math.round(Math.max(-2, Math.min(2, v)) * 10) / 10;
    return [snap(wx), snap(wy)];
  }

  function onMouseDown(e) {
    const [px, py] = canvasPos(e);
    const [wx, wy] = canvasToWorld(px, py, hoverCanvas.width, hoverCanvas.height);
    const snap = v => Math.round(Math.max(-2, Math.min(2, v)) * 10) / 10;
    dragOriginCanvas = { px, py };
    dragOriginWorld  = { wx: snap(wx), wy: snap(wy) };
    isDragging = false;
  }

  function onMouseMove(e) {
    if (dragOriginCanvas) {
      const [px, py] = canvasPos(e);
      if (Math.hypot(px - dragOriginCanvas.px, py - dragOriginCanvas.py) > 5) {
        isDragging = true;
        const [wx, wy] = canvasToWorld(px, py, hoverCanvas.width, hoverCanvas.height);
        const snap = v => Math.round(Math.max(-2, Math.min(2, v)) * 10) / 10;
        drawVectorDrag(hoverCanvas, dragOriginWorld.wx, dragOriginWorld.wy, snap(wx), snap(wy));
        return;
      }
    }
    const [wx, wy] = getSnapped(e);
    drawHoverMarker(hoverCanvas, wx, wy);
  }

  function onMouseUp(e) {
    if (isDragging && dragOriginWorld) {
      const [wx, wy] = getSnapped(e);
      // Swap to UI convention: UI X = horizontal = robot Y (wy), UI Y = vertical = robot X (wx)
      dispatch('gridDrag', { dx: wy - dragOriginWorld.wy, dy: wx - dragOriginWorld.wx });
    } else if (dragOriginWorld) {
      const [wx, wy] = getSnapped(e);
      dispatch('gridClick', { x: wy, y: wx });
    }
    dragOriginCanvas = null;
    dragOriginWorld  = null;
    isDragging       = false;
  }

  function onMouseLeave() {
    dragOriginCanvas = null;
    dragOriginWorld  = null;
    isDragging       = false;
    hoverCanvas.getContext('2d').clearRect(0, 0, hoverCanvas.width, hoverCanvas.height);
  }

  $: scoreClass = score == null ? '' : score.score >= 70 ? 'good' : score.score >= 40 ? 'ok' : 'poor';

  $: if (targetCanvas) drawCanvasTrail(targetCanvas, targetTrail);
  $: if (trailCanvas)  { drawCanvasTrail(trailCanvas, studentTrail); drawPenMarkers(trailCanvas, studentPenEvents); }

  onMount(() => {
    const panelRo = new ResizeObserver(([entry]) => { panelWidth = entry.contentRect.width; });
    panelRo.observe(el);

    const frameRo = new ResizeObserver(([entry]) => {
      const fw = entry.contentRect.width;
      canvasSize = Math.min(220, Math.floor((fw - 8) / 2));
    });
    frameRo.observe(frameImg);

    const updateLabelPad = () => {
      targetLabelPad = Math.max(0, trailLabel.scrollHeight - targetLabel.scrollHeight);
    };
    const labelRo = new ResizeObserver(updateLabelPad);
    labelRo.observe(trailLabel);
    requestAnimationFrame(updateLabelPad);

    return () => { panelRo.disconnect(); frameRo.disconnect(); labelRo.disconnect(); };
  });
</script>

<div id="output-panel" bind:this={el}>
  <span class="panel-label">// 3D VIEW</span>
  <div id="view-grid">
    <button class="nav-btn nav-prev" disabled={!canGoPrev} on:click={() => dispatch('prevFrame')}>&lt;</button>
    <img id="frame-display" class="retro-border" bind:this={frameImg} src={frameDataUrl || undefined} alt="Robot view" />
    <button class="nav-btn nav-next" disabled={!canGoNext} on:click={() => dispatch('nextFrame')}>&gt;</button>
    <div id="frame-info">{frameInfo}</div>
    <div id="canvas-row">
      <div class="canvas-col">
        <span class="panel-label" bind:this={targetLabel} style="padding-top:{targetLabelPad}px">// TARGET</span>
        <div class="canvas-wrapper">
          <canvas bind:this={targetCanvas} class="retro-border" width="220" height="220" style="width:{canvasSize}px;height:{canvasSize}px"></canvas>
          <canvas bind:this={hoverCanvas} class="hover-overlay" width="220" height="220"
            on:mousedown={onMouseDown}
            on:mousemove={onMouseMove}
            on:mouseup={onMouseUp}
            on:mouseleave={onMouseLeave}
          ></canvas>
        </div>
      </div>
      <div class="canvas-col">
        <span class="panel-label" bind:this={trailLabel}>// YOUR DRAWING</span>
        <canvas bind:this={trailCanvas} class="retro-border" width="220" height="220" style="width:{canvasSize}px;height:{canvasSize}px"></canvas>
      </div>
    </div>
  </div>

  {#if score != null}
    <div id="score-box" class="retro-border {scoreClass}">
      SCORE: {score.score} / 100
      <div id="score-detail">
        Coverage {score.breakdown.coverage}% // Precision {score.breakdown.precision}% // Colour {score.breakdown.color}%
      </div>
    </div>
  {/if}
</div>

<style>
  #output-panel {
    flex: 1; background: var(--panel);
    display: flex; flex-direction: column;
    align-items: center; padding: 10px; gap: 8px;
    overflow-y: auto; border-left: var(--px) solid var(--border);
  }
  .panel-label {
    font-size: 22px; color: var(--highlight); text-transform: uppercase;
    letter-spacing: 1px; text-shadow: 1px 1px 0 rgba(0,0,0,.5); text-align: center;
  }
  #view-grid     { display: grid; grid-template-columns: auto auto auto; max-width: 100%; row-gap: 8px; }
  .nav-prev      { grid-column: 1; grid-row: 1; align-self: center; }
  #frame-display { grid-column: 2; grid-row: 1; max-width: 100%; max-height: 38vh; background: #000; object-fit: contain; }
  .nav-next      { grid-column: 3; grid-row: 1; align-self: center; }
  #frame-info    { grid-column: 2; grid-row: 2; font-size: 22px; color: var(--muted); text-align: center; }
  #canvas-row    { grid-column: 2; grid-row: 3; display: flex; justify-content: center; gap: 8px; align-items: flex-start; }
  .nav-btn {
    background: none; border: none; cursor: pointer;
    font-family: 'Silkscreen', monospace; font-size: 32px; line-height: 1;
    color: #7c3aed; padding: 2px 6px; transition: color 0.1s;
  }
  .nav-btn:disabled { color: #2e1a4a; cursor: default; }
  .nav-btn:not(:disabled):hover { color: #a78bfa; }
  .canvas-col    { display: flex; flex-direction: column; align-items: center; gap: 4px; }
  .canvas-wrapper { position: relative; }
  .hover-overlay  { position: absolute; inset: 0; width: 100%; height: 100%; cursor: crosshair; }
  #score-box     { text-align: center; padding: 8px 14px; font-size: 32px; }
  #score-box.good { background: var(--accent); color: white; box-shadow: var(--px) var(--px) 0 #5b21b6; }
  #score-box.ok   { background: #7c3aed;       color: white; box-shadow: var(--px) var(--px) 0 #5b21b6; }
  #score-box.poor { background: #3b1768; color: var(--highlight); box-shadow: var(--px) var(--px) 0 #2e1065; }
  #score-detail   { font-size: 20px; font-weight: 400; margin-top: 4px; opacity: 0.8; }
</style>
