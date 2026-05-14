<script>
  import { onMount, createEventDispatcher } from 'svelte';
  import TamaBot from './TamaBot.svelte';
  import { drawCanvasTrail, drawPaintBuckets, drawPenMarkers, canvasToWorld, drawHoverMarker, drawVectorDrag } from './canvas.js';

  const dispatch = createEventDispatcher();

  export let frameDataUrl = '';
  export let frameInfo = 'DRAG BLOCKS AND CLICK RUN';
  export let frameLabel = null;
  export let studentTrail = [];
  export let targetTrail = [];
  export let score = null;
  export let studentPenEvents = [];
  export let paintBuckets = [];
  export let visitedBuckets = [];
  export let canGoPrev = false;
  export let canGoNext = false;
  export let showTarget = true;
  export let tamaMsg = '';
  export let tamaVisible = false;
  export let tamaIsError = false;
  export let tamaIsWarning = false;
  export let onTamaPoke = () => {};

  let el;
  let scrollEl;
  let scoreEl;
  let frameImg;
  let targetCanvas;
  let hoverCanvas;
  let trailCanvas;
  let hoverTrailCanvas;
  let canvasSize = 220;

  $: if (score != null && scoreEl) requestAnimationFrame(() => scoreEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' }));

  function makeInteraction(getCanvas) {
    let originCanvas = null, originWorld = null, dragging = false;
    const snap = v => Math.round(Math.max(-2, Math.min(2, v)) * 10) / 10;
    function pos(e) {
      const cvs = getCanvas();
      const r = cvs.getBoundingClientRect();
      return [(e.clientX - r.left) * cvs.width / r.width,
              (e.clientY - r.top)  * cvs.height / r.height];
    }
    function snapped(e) {
      const cvs = getCanvas();
      const [px, py] = pos(e);
      const [wx, wy] = canvasToWorld(px, py, cvs.width, cvs.height);
      return [snap(wx), snap(wy)];
    }
    return {
      down(e) {
        const cvs = getCanvas();
        const [px, py] = pos(e);
        const [wx, wy] = canvasToWorld(px, py, cvs.width, cvs.height);
        originCanvas = { px, py };
        originWorld  = { wx: snap(wx), wy: snap(wy) };
        dragging = false;
      },
      move(e) {
        const cvs = getCanvas();
        if (originCanvas) {
          const [px, py] = pos(e);
          if (Math.hypot(px - originCanvas.px, py - originCanvas.py) > 5) {
            dragging = true;
            const [wx, wy] = canvasToWorld(px, py, cvs.width, cvs.height);
            drawVectorDrag(cvs, originWorld.wx, originWorld.wy, snap(wx), snap(wy));
            return;
          }
        }
        const [wx, wy] = snapped(e);
        drawHoverMarker(cvs, wx, wy);
      },
      up(e) {
        if (dragging && originWorld) {
          const [wx, wy] = snapped(e);
          dispatch('gridDrag', { dx: wy - originWorld.wy, dy: wx - originWorld.wx });
        } else if (originWorld) {
          const [wx, wy] = snapped(e);
          dispatch('gridClick', { x: wy, y: wx });
        }
        originCanvas = null; originWorld = null; dragging = false;
      },
      leave() {
        originCanvas = null; originWorld = null; dragging = false;
        const cvs = getCanvas();
        cvs.getContext('2d').clearRect(0, 0, cvs.width, cvs.height);
      },
    };
  }

  const ti = makeInteraction(() => hoverCanvas);
  const di = makeInteraction(() => hoverTrailCanvas);

  $: scoreClass = score == null ? '' : score.score >= 70 ? 'good' : score.score >= 40 ? 'ok' : 'poor';

  $: if (targetCanvas) { drawCanvasTrail(targetCanvas, targetTrail); drawPaintBuckets(targetCanvas, paintBuckets, []); }
  $: if (trailCanvas)  { drawCanvasTrail(trailCanvas, studentTrail); drawPaintBuckets(trailCanvas, paintBuckets, visitedBuckets); drawPenMarkers(trailCanvas, studentPenEvents); }

  onMount(() => {
    const ro = new ResizeObserver(([entry]) => {
      const pw = entry.contentRect.width;
      canvasSize = Math.min(260, Math.floor((pw - 40) / 2));
    });
    ro.observe(el);

    return () => { ro.disconnect(); };
  });
</script>

<div id="output-panel" bind:this={el}>
  <div id="output-scroll" bind:this={scrollEl}>
    <span class="panel-label">// 3D VIEW</span>
    <div id="view-grid">
      <button class="nav-btn nav-prev" disabled={!canGoPrev} on:click={() => dispatch('prevFrame')}>&lt;</button>
      <div id="frame-wrap">
        <img id="frame-display" class="retro-border" bind:this={frameImg} src={frameDataUrl || undefined} alt="Robot view" />
        {#if frameLabel}
          <div class="frame-label" style="color:rgb({frameLabel.r},{frameLabel.g},{frameLabel.b});text-shadow:0 0 10px rgb({frameLabel.r},{frameLabel.g},{frameLabel.b})">
            {frameLabel.text}
          </div>
        {/if}
      </div>
      <button class="nav-btn nav-next" disabled={!canGoNext} on:click={() => dispatch('nextFrame')}>&gt;</button>
      <div id="frame-info">{frameInfo}</div>
      <div id="canvas-row">
        {#if showTarget}
        <div class="canvas-col">
          <span class="canvas-label">// TARGET</span>
          <div class="canvas-wrapper">
            <canvas bind:this={targetCanvas} class="retro-border" width="220" height="220" style="width:{canvasSize}px;height:{canvasSize}px"></canvas>
            <canvas bind:this={hoverCanvas} class="hover-overlay" width="220" height="220"
              on:mousedown={ti.down} on:mousemove={ti.move}
              on:mouseup={ti.up} on:mouseleave={ti.leave}
            ></canvas>
          </div>
        </div>
        {/if}
        <div class="canvas-col">
          <span class="canvas-label">// YOURS</span>
          <div class="canvas-wrapper">
            <canvas bind:this={trailCanvas} class="retro-border" width="220" height="220" style="width:{canvasSize}px;height:{canvasSize}px"></canvas>
            <canvas bind:this={hoverTrailCanvas} class="hover-overlay" width="220" height="220"
              on:mousedown={di.down} on:mousemove={di.move}
              on:mouseup={di.up} on:mouseleave={di.leave}
            ></canvas>
          </div>
        </div>
      </div>
    </div>

    {#if score != null}
      <div id="score-box" class="retro-border {scoreClass}" bind:this={scoreEl}>
        SCORE: {score.score} / 100
        <div id="score-detail">
          Coverage {score.breakdown.coverage}% // Precision {score.breakdown.precision}% // Colour {score.breakdown.color}%
        </div>
      </div>
    {/if}

    <div id="tama-area">
      <TamaBot message={tamaMsg} visible={tamaVisible} isError={tamaIsError} isWarning={tamaIsWarning} onPoke={onTamaPoke} />
    </div>
  </div>
</div>

<style>
  #output-panel {
    width: 520px; flex-shrink: 0;
    background: var(--panel);
    display: flex; flex-direction: column;
    overflow: hidden;
    border-left: var(--px) solid var(--border);
  }
  #output-scroll {
    flex: 1; overflow-y: auto;
    display: flex; flex-direction: column;
    align-items: center; padding: 10px; gap: 8px;
  }
  .panel-label {
    font-size: 22px; color: var(--highlight); text-transform: uppercase;
    letter-spacing: 1px; text-shadow: 1px 1px 0 rgba(0,0,0,.5); text-align: center;
  }
  .canvas-label {
    font-size: 18px; color: var(--highlight); text-transform: uppercase;
    letter-spacing: 1px; text-shadow: 1px 1px 0 rgba(0,0,0,.5);
    text-align: center; white-space: nowrap;
  }
  #view-grid     { display: grid; grid-template-columns: auto 1fr auto; width: 100%; row-gap: 8px; }
  .nav-prev      { grid-column: 1; grid-row: 1; align-self: center; }
  #frame-wrap    { grid-column: 2; grid-row: 1; position: relative; display: flex; min-width: 0; }
  #frame-display { max-width: 100%; max-height: 36vh; background: #000; object-fit: contain; width: 100%; }
  .frame-label   {
    position: absolute; bottom: 6px; left: 50%; transform: translateX(-50%);
    font-family: 'Silkscreen', monospace; font-size: 18px; white-space: nowrap;
    background: rgba(0,0,0,0.55); padding: 3px 10px; pointer-events: none;
  }
  .nav-next      { grid-column: 3; grid-row: 1; align-self: center; }
  #frame-info    { grid-column: 2; grid-row: 2; font-size: 22px; color: var(--muted); text-align: center; }
  #canvas-row    { grid-column: 1 / -1; grid-row: 3; display: flex; justify-content: center; gap: 10px; align-items: flex-start; }
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
  #tama-area     { width: 100%; margin-top: auto; }
</style>
