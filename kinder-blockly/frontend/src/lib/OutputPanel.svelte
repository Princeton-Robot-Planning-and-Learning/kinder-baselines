<script>
  import { onMount } from 'svelte';
  import { drawCanvasTrail } from './canvas.js';

  export let frameDataUrl = '';
  export let frameInfo = 'DRAG BLOCKS AND CLICK RUN';
  export let studentTrail = [];
  export let targetTrail = [];
  export let score = null;
  export let panelWidth = 0;

  let el;
  let targetCanvas;
  let trailCanvas;

  $: scoreClass = score == null ? '' : score.score >= 70 ? 'good' : score.score >= 40 ? 'ok' : 'poor';

  $: if (targetCanvas) drawCanvasTrail(targetCanvas, targetTrail);
  $: if (trailCanvas)  drawCanvasTrail(trailCanvas,  studentTrail);

  onMount(() => {
    const ro = new ResizeObserver(([entry]) => { panelWidth = entry.contentRect.width; });
    ro.observe(el);
    return () => ro.disconnect();
  });
</script>

<div id="output-panel" bind:this={el}>
  <span class="panel-label">// 3D VIEW</span>
  <img id="frame-display" class="retro-border" src={frameDataUrl || undefined} alt="Robot view" />
  <div id="frame-info">{frameInfo}</div>

  <div id="canvas-row">
    <div class="canvas-col">
      <span class="panel-label">// TARGET</span>
      <canvas bind:this={targetCanvas} class="retro-border" width="220" height="220"></canvas>
    </div>
    <div class="canvas-col">
      <span class="panel-label">// YOUR DRAWING</span>
      <canvas bind:this={trailCanvas} class="retro-border" width="220" height="220"></canvas>
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
    letter-spacing: 1px; text-shadow: 1px 1px 0 rgba(0,0,0,.5);
  }
  #frame-display { max-width: 100%; max-height: 38%; background: #000; }
  #frame-info    { font-size: 22px; color: var(--muted); }
  #canvas-row    { display: flex; gap: 8px; align-items: flex-start; }
  .canvas-col    { display: flex; flex-direction: column; align-items: center; gap: 4px; }
  #score-box     { text-align: center; padding: 8px 14px; font-size: 32px; }
  #score-box.good { background: var(--accent); color: white; box-shadow: var(--px) var(--px) 0 #5b21b6; }
  #score-box.ok   { background: #7c3aed;       color: white; box-shadow: var(--px) var(--px) 0 #5b21b6; }
  #score-box.poor { background: #3b1768; color: var(--highlight); box-shadow: var(--px) var(--px) 0 #2e1065; }
  #score-detail   { font-size: 20px; font-weight: 400; margin-top: 4px; opacity: 0.8; }
</style>
