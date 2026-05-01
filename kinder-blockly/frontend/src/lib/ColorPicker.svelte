<script>
  import { createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher();

  export let x = 0;
  export let y = 0;
  export let selectedIdx = 0; // which PRESETS index is currently active

  const PRESETS = [
    [255,   0,   0],  // 0 RED
    [255, 140,   0],  // 1 ORANGE
    [220, 180,   0],  // 2 YELLOW
    [  0, 200,   0],  // 3 GREEN
    [  0, 200, 200],  // 4 CYAN
    [  0,  80, 255],  // 5 BLUE
    [160,   0, 240],  // 6 VIOLET
  ];

  // SVG geometry — pointy-top hexagons, circumradius R=36
  const R = 36;
  const D = R * Math.sqrt(3);
  const SW = 230;
  const SH = 215;
  const CX = SW / 2;
  const CY = SH / 2;

  const SURROUND_ANGLES = [0, 60, 120, 180, 240, 300].map(a => a * Math.PI / 180);
  const HEX_POSITIONS = [
    [CX, CY],
    ...SURROUND_ANGLES.map(a => [CX + D * Math.cos(a), CY + D * Math.sin(a)])
  ];

  const HEX_VERTICES = Array.from({length: 6}, (_, i) => {
    const a = (i * 60 - 30) * Math.PI / 180;
    return [R * Math.cos(a), R * Math.sin(a)];
  });

  function hexPath(cx, cy) {
    return HEX_VERTICES.map(([vx, vy], i) =>
      (i === 0 ? 'M' : 'L') + (cx + vx).toFixed(2) + ',' + (cy + vy).toFixed(2)
    ).join(' ') + ' Z';
  }

  function toHex(r, g, b) {
    return '#' + [r, g, b].map(v => v.toString(16).padStart(2, '0')).join('');
  }

  function selectColor(idx) {
    const [r, g, b] = PRESETS[idx];
    dispatch('pick', { r, g, b, idx });
  }

  function onBackdrop() {
    dispatch('close');
  }
</script>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<!-- svelte-ignore a11y-no-static-element-interactions -->
<div class="backdrop" on:click={onBackdrop}></div>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<!-- svelte-ignore a11y-no-static-element-interactions -->
<div class="picker" style="left:{x}px;top:{y}px" on:click|stopPropagation>
  <svg width={SW} height={SH} xmlns="http://www.w3.org/2000/svg">
    {#each HEX_POSITIONS as [hx, hy], i}
      {#if i !== selectedIdx}
        {@const [r, g, b] = PRESETS[i]}
        <!-- svelte-ignore a11y-click-events-have-key-events -->
        <!-- svelte-ignore a11y-no-static-element-interactions -->
        <path
          d={hexPath(hx, hy)}
          fill={toHex(r, g, b)}
          class="hex hex-idle"
          stroke="#3d1560"
          stroke-width="2"
          on:click={() => selectColor(i)}
        />
      {/if}
    {/each}
    {#each [selectedIdx] as si}
      {@const [sr, sg, sb] = PRESETS[si]}
      {@const [shx, shy] = HEX_POSITIONS[si]}
      <!-- svelte-ignore a11y-click-events-have-key-events -->
      <!-- svelte-ignore a11y-no-static-element-interactions -->
      <path
        d={hexPath(shx, shy)}
        fill={toHex(sr, sg, sb)}
        class="hex hex-selected"
        stroke="#ffffff"
        stroke-width="3"
        on:click={() => selectColor(si)}
      />
    {/each}
  </svg>
</div>

<style>
  .backdrop {
    position: fixed;
    inset: 0;
    z-index: 9998;
    background: transparent;
  }

  .picker {
    position: fixed;
    z-index: 9999;
    background: #06020f;
    border: 3px solid #7c3aed;
    border-radius: 4px;
    box-shadow: 0 0 24px rgba(124, 58, 237, 0.6), 0 0 8px rgba(124, 58, 237, 0.4);
    padding: 8px;
  }

  .hex {
    cursor: pointer;
    transition: filter 0.1s, stroke 0.1s;
  }

  .hex-selected {
    filter: drop-shadow(0 0 8px #ffffff);
  }

  .hex-idle:hover {
    stroke: #a78bfa;
    filter: drop-shadow(0 0 6px #a78bfa) brightness(1.3);
  }
</style>
