<script>
  import { createEventDispatcher } from 'svelte';
  export let challenges = [];
  export let isRunning = false;
  export let status = '';
  export let hint = '';

  const dispatch = createEventDispatcher();
  let selectedId = '';

  function onChallengeChange() { dispatch('challengeChange', selectedId); }
</script>

<header>
  <h1>KinDER Blockly</h1>
  <button id="run-btn" on:click={() => dispatch('run')} disabled={isRunning}>&#9654; RUN</button>
  <select id="challenge-select" bind:value={selectedId} on:change={onChallengeChange}>
    <option value="">FREE DRAW</option>
    {#each challenges as c}
      {@const stars = c.difficulty === 'easy' ? '*' : c.difficulty === 'medium' ? '**' : '***'}
      <option value={c.id}>[{stars}] {c.name.toUpperCase()}</option>
    {/each}
  </select>
  <span id="challenge-hint">{hint}</span>
  <span id="status">{status}</span>
</header>

<style>
  header {
    background: var(--panel); padding: 10px 16px;
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    border-bottom: var(--px) solid var(--border);
  }
  h1 { font-size: 36px; color: var(--accent); text-shadow: 2px 2px 0 #2e1065; }
  #run-btn {
    font-family: 'Silkscreen', monospace;
    background: var(--accent); color: white;
    border: var(--px) solid #7c3aed;
    padding: 8px 18px; font-size: 26px; cursor: pointer;
    box-shadow: var(--px) var(--px) 0 #4c1d95;
    transition: transform 0.05s;
  }
  #run-btn:hover  { transform: translate(-1px, -1px); }
  #run-btn:active { transform: translate(var(--px), var(--px)); box-shadow: none; }
  #run-btn:disabled { background: #2a1d4e; border-color: #1e1040; box-shadow: none; cursor: not-allowed; }
  #challenge-select {
    font-family: 'Silkscreen', monospace;
    padding: 6px 8px; font-size: 24px;
    border: var(--px) solid var(--border);
    background: var(--surface); color: var(--highlight); cursor: pointer;
  }
  #challenge-hint { font-size: 22px; color: var(--muted); }
  #status         { font-size: 22px; color: var(--highlight); }
</style>
