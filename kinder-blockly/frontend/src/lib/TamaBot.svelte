<script>
  export let message = '';
  export let visible = false;
  export let isError = false;
  export let onPoke = () => {};
</script>

<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
<div id="tama-container">
  <div id="tama-bubble" class:visible class:error={isError} on:click={onPoke}>{message}</div>
  <div id="tama-bot" on:click={onPoke} title="Click me!">
    <div id="tama-pixel"></div>
  </div>
</div>

<style>
  #tama-container {
    display: flex; flex-direction: row; align-items: flex-end; gap: 6px;
    padding: 8px 10px 10px;
    pointer-events: none;
    width: 100%;
    box-sizing: border-box;
  }
  #tama-bubble {
    flex: 1; min-width: 0;
    background: var(--surface); color: var(--text);
    border: var(--px) solid var(--accent);
    box-shadow: var(--px) var(--px) 0 rgba(145,96,238,.35);
    padding: 10px 14px; font-size: 20px; line-height: 1.5;
    position: relative;
    opacity: 0; transform: translateX(-6px);
    transition: opacity 0.3s, transform 0.3s;
    pointer-events: auto; cursor: pointer;
    box-sizing: border-box;
  }
  #tama-bubble.visible { opacity: 1; transform: translateX(0); }
  #tama-bubble.error {
    border-color: #ef4444;
    box-shadow: var(--px) var(--px) 0 rgba(239,68,68,.4);
    background: #1a0505;
    color: #fca5a5;
  }
  #tama-bubble::after {
    content: ''; position: absolute; right: -8px; bottom: 28px;
    width: 0; height: 0;
    border-top: 8px solid transparent; border-bottom: 8px solid transparent;
    border-left: 8px solid var(--accent);
  }
  #tama-bubble.error::after { border-left-color: #ef4444; }
  #tama-bot {
    flex-shrink: 0;
    width: 96px; height: 108px; position: relative;
    pointer-events: auto; cursor: pointer;
    animation: tama-bounce 1.2s ease-in-out infinite;
  }
  @keyframes tama-bounce {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(-6px); }
  }
  #tama-pixel {
    position: absolute; top: 0; left: 0;
    width: 6px; height: 6px; background: transparent;
    box-shadow:
      36px 0px var(--gold), 54px 0px var(--gold),
      30px 6px var(--text), 36px 6px var(--highlight),
      42px 6px var(--pink), 48px 6px var(--highlight), 54px 6px var(--text),
      42px 12px var(--muted), 48px 12px var(--muted),
      36px 18px var(--text), 42px 18px var(--text), 48px 18px var(--text), 54px 18px var(--text),
      36px 24px var(--text), 42px 24px var(--text), 48px 24px var(--text), 54px 24px var(--text),
      24px 30px var(--accent), 30px 30px var(--accent), 36px 30px var(--accent), 42px 30px var(--accent),
      48px 30px var(--accent), 54px 30px var(--accent), 60px 30px var(--accent),
      42px 36px var(--text), 48px 36px var(--text),
      42px 42px var(--text), 48px 42px var(--text),
      42px 48px var(--text), 48px 48px var(--text),
      24px 54px var(--accent), 30px 54px var(--accent), 36px 54px var(--accent), 42px 54px var(--accent),
      48px 54px var(--accent), 54px 54px var(--accent), 60px 54px var(--accent),
      18px 60px var(--muted), 24px 60px var(--muted), 30px 60px var(--muted), 36px 60px var(--muted),
      42px 60px var(--muted), 48px 60px var(--muted), 54px 60px var(--muted), 60px 60px var(--muted), 66px 60px var(--muted),
      12px 66px var(--border), 18px 66px var(--border), 24px 66px var(--border), 30px 66px var(--border),
      36px 66px var(--border), 42px 66px var(--border), 48px 66px var(--border), 54px 66px var(--border),
      60px 66px var(--border), 66px 66px var(--border), 72px 66px var(--border);
  }
</style>
