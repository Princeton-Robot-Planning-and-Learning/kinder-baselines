<script>
  import { onMount, onDestroy } from 'svelte';
  import Header from './lib/Header.svelte';
  import BlocklyWorkspace from './lib/BlocklyWorkspace.svelte';
  import OutputPanel from './lib/OutputPanel.svelte';

  let blocklyWorkspace;

  let challenges = [];
  let currentChallenge = null;
  let isRunning = false;
  let isStopped = false;
  let runAbortController = null;
  let status = '';
  let allFrames = [];
  let allFrameLabels = [];
  let currentFrameIndex = -1;
  let frameInfo = 'DRAG BLOCKS AND CLICK RUN';

  let lastFrameDataUrl = '';
  $: frameDataUrl = currentFrameIndex >= 0 && allFrames.length > 0
    ? 'data:image/jpeg;base64,' + allFrames[currentFrameIndex]
    : lastFrameDataUrl;
  $: frameLabel = allFrameLabels[currentFrameIndex] ?? null;
  $: canGoPrev = currentFrameIndex > 0;
  $: canGoNext = currentFrameIndex >= 0 && currentFrameIndex < allFrames.length - 1;

  function prevFrame() {
    if (canGoPrev) { currentFrameIndex--; frameInfo = `FRAME ${currentFrameIndex + 1}/${allFrames.length}`; }
  }
  function nextFrame() {
    if (canGoNext) { currentFrameIndex++; frameInfo = `FRAME ${currentFrameIndex + 1}/${allFrames.length}`; }
  }
  let studentTrail = [];
  let studentPenEvents = [];
  let targetTrail = [];
  let paintBuckets = [];
  let spawnedBuckets = [];
  let visitedBuckets = [];
  $: allPaintBuckets = [...paintBuckets, ...spawnedBuckets];
  let scoreData = null;
  let tamaMsg = '';
  let tamaVisible = false;
  let tamaIsError = false;
  let tamaTimer = null;

  const TAMA_IDLE_TIPS = [
    "I YEARN to draw... please, give me blocks!",
    "TIP: PEN UP lets me glide without leaving a mark. Stealth mode.",
    "TIP: You can change colours mid-stroke! I contain multitudes.",
    "TIP: I start at (0, 0). Humble beginnings for a great artist.",
    "Every masterpiece begins with a single block...",
    "HINT: The Target canvas haunts my dreams. I MUST replicate it.",
    "HINT: Try simple shapes first! Even Picasso started somewhere.",
    "TIP: X and Y go from -2 to 2. That's my whole world. It's enough.",
    "Click me! I'm lonely and full of wisdom!",
    "TIP: X goes up, Y goes right. I didn't make the rules.",
    "I was BORN to draw. Tidying was just my day job.",
    "A robot without a pen is like a bird without wings. Tragic.",
    "They said I could be anything... so I became an ARTIST.",
    "My circuits tingle when you drag blocks. Keep going!",
    "One does not simply free draw. One EXPRESSES oneself.",
    "HINT: Check the Target canvas for your goal!",
    "TIP: Click me any time for a hint...",
  ];

  function tamaSay(msg, duration = 5000) {
    tamaMsg = msg; tamaVisible = true; tamaIsError = false;
    if (tamaTimer) clearTimeout(tamaTimer);
    tamaTimer = setTimeout(() => { tamaVisible = false; }, duration);
  }

  function tamaSayError(msg, duration = 6000) {
    tamaMsg = msg; tamaVisible = true; tamaIsError = true;
    if (tamaTimer) clearTimeout(tamaTimer);
    tamaTimer = setTimeout(() => { tamaVisible = false; tamaIsError = false; }, duration);
  }

  function tamaPoke() {
    tamaSay(TAMA_IDLE_TIPS[Math.floor(Math.random() * TAMA_IDLE_TIPS.length)], 4000);
  }

  const onKey = e => { if (e.key === 'Escape') { tamaVisible = false; tamaIsError = false; if (tamaTimer) clearTimeout(tamaTimer); } };
  onDestroy(() => document.removeEventListener('keydown', onKey));

  onMount(async () => {
    document.addEventListener('keydown', onKey);

    await Promise.all([loadChallenges(), loadInitialFrame()]);
    setTimeout(() => tamaSay(
      "Greetings, young artist! I am TidyBot, and I LIVE to draw. Pick a challenge... or let me free draw. Please.",
      6000
    ), 1500);
  });

  async function loadChallenges() {
    try {
      const r = await fetch('/challenges');
      const d = await r.json();
      challenges = d.challenges || [];
    } catch {}
  }

  async function loadInitialFrame(buckets = []) {
    frameInfo = 'LOADING WORLD...';
    try {
      const r = await fetch('/reset', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({seed:0, paint_buckets: buckets}) });
      const d = await r.json();
      if (d.frame) { allFrames = [d.frame]; currentFrameIndex = 0; frameInfo = 'READY!'; }
    } catch { frameInfo = 'LOAD FAILED'; }
  }

  async function onChallengeChange(id) {
    // Clear everything tied to the previous challenge's run so the YOURS
    // canvas, 3D frame label, and score do not bleed across challenges.
    scoreData = null;
    studentTrail = []; studentPenEvents = [];
    spawnedBuckets = []; allFrameLabels = [];
    if (!id) {
      currentChallenge = null; targetTrail = []; paintBuckets = []; visitedBuckets = [];
      blocklyWorkspace.setPenColorEnabled(true);
      tamaSay("FREE DRAW! No rules, no limits! Just me and the canvas. *chef's kiss*", 4000);
      await loadInitialFrame([]);
      return;
    }
    try {
      const r = await fetch('/challenges/' + id);
      const c = await r.json();
      currentChallenge = c; targetTrail = c.target_trail || [];
      paintBuckets = c.paint_buckets || []; visitedBuckets = [];
      blocklyWorkspace.setPenColorEnabled((c.paint_buckets?.length ?? 0) === 0);
      tamaSay(c.hint || c.description || 'Good luck!', 5000);
      await loadInitialFrame(paintBuckets);
    } catch { tamaSay("Failed to load challenge!", 4000); }
  }

  async function stopProgram() {
    isStopped = true;
    runAbortController?.abort();
    fetch('/stop', { method: 'POST' }).catch(() => {});
    tamaSay("OK OK, brakes applied! I'll stop after this move.", 4000);
  }

  async function runProgram() {
    if (!blocklyWorkspace.hasStartBlock()) { status = 'NO START!'; tamaSayError("I need a Start block! Drag one from the Program category.", 5000); return; }
    const program = blocklyWorkspace.getProgram();
    if (program.blocks.length === 0) { status = 'NO BLOCKS!'; tamaSayError("Connect some blocks under the Start block first!", 4000); return; }
    const paramErr = blocklyWorkspace.hasParamErrors();
    if (paramErr) { status = 'ERRORS!'; tamaSayError(paramErr, 6000); return; }

    isStopped = false;
    runAbortController = new AbortController();
    isRunning = true; status = 'RUNNING...'; frameInfo = 'STARTING...';
    scoreData = null; studentTrail = []; studentPenEvents = []; visitedBuckets = []; spawnedBuckets = [];
    if (allFrames.length > 0 && currentFrameIndex >= 0) lastFrameDataUrl = 'data:image/jpeg;base64,' + allFrames[currentFrameIndex];
    allFrames = []; allFrameLabels = [];
    tamaSay("Let's go! Executing program...", 3000);

    try {
      const r = await fetch('/run', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ program, seed: 0, paint_buckets: paintBuckets }),
        signal: runAbortController.signal,
      });

      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      let doneMsg = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop(); // keep trailing incomplete line
        for (const line of lines) {
          if (!line.trim()) continue;
          const msg = JSON.parse(line);
          if (msg.type === 'frame') {
            allFrames = [...allFrames, msg.frame];
            allFrameLabels = [...allFrameLabels, msg.label];
            currentFrameIndex = allFrames.length - 1;
            const n = allFrames.length;
            frameInfo = `FRAME ${n}`;
            status = `RUNNING... (frame ${n})`;
          } else if (msg.type === 'done') {
            doneMsg = msg;
          }
        }
      }

      if (doneMsg) {
        if (doneMsg.error_block_id) blocklyWorkspace.selectBlock(doneMsg.error_block_id);

        if (doneMsg.error) {
          status = 'ERROR!';
          tamaSayError(doneMsg.error, 7000);
        } else if (doneMsg.infinite_loop) {
          status = 'LOOP!';
          tamaSayError("I'm going in circles!! My while loop ran 100 times and never stopped — I think I'm stuck forever. Could you check that condition?", 7000);
        } else if (!isStopped) {
          status = 'DONE!';
        }

        if (!allFrames.length) {
          frameInfo = doneMsg.error ? 'CHECK BLOCKS' : (doneMsg.infinite_loop ? 'INFINITE LOOP' : 'NO FRAMES');
        } else {
          frameInfo = `FRAME ${currentFrameIndex + 1}/${allFrames.length}`;
        }

        studentTrail = doneMsg.trail || [];
        studentPenEvents = doneMsg.pen_events || [];
        visitedBuckets = doneMsg.visited_buckets || [];
        spawnedBuckets = doneMsg.spawned_buckets || [];

        if (currentChallenge && studentTrail.length > 0) {
          await requestScore(currentChallenge.id, studentTrail);
        } else if (!currentChallenge && !doneMsg.error && !doneMsg.infinite_loop && !isStopped) {
          tamaSay("Nice drawing! Pick a challenge to get scored!", 4000);
        }
      }
    } catch (e) {
      if (e?.name === 'AbortError') {
        status = 'STOPPED';
        frameInfo = allFrames.length ? `FRAME ${currentFrameIndex + 1}/${allFrames.length}` : 'STOPPED';
      } else { status = 'ERROR!'; tamaSayError("Connection error! Is the server running?", 4000); }
    }
    finally { isRunning = false; }
  }

  async function requestScore(challengeId, trail) {
    try {
      const r = await fetch('/score', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ challenge_id: challengeId, student_trail: trail }),
      });
      scoreData = await r.json();
      const s = scoreData.score;
      if      (s >= 90) tamaSay("PERFECT! You're a robot artist!", 5000);
      else if (s >= 70) tamaSay("Great job! Almost perfect!", 4000);
      else if (s >= 40) tamaSay("Getting there! Check the target and try again!", 5000);
      else              tamaSay("Keep trying! Compare your drawing to the target.", 5000);
    } catch (e) { console.error('Score request failed:', e); }
  }
</script>

<Header {challenges} {isRunning} {status}
  on:run={runProgram}
  on:stop={stopProgram}
  on:challengeChange={e => onChallengeChange(e.detail)}
/>

<div class="content">
  <BlocklyWorkspace bind:this={blocklyWorkspace} on:message={e => tamaSayError(e.detail, 4000)} />
  <OutputPanel
    {frameDataUrl} {frameInfo} {frameLabel} {studentTrail} {studentPenEvents} {targetTrail} score={scoreData}
    paintBuckets={allPaintBuckets} {visitedBuckets}
    showTarget={currentChallenge !== null}
    {canGoPrev} {canGoNext}
    tamaMsg={tamaMsg} tamaVisible={tamaVisible} {tamaIsError} onTamaPoke={tamaPoke}
    on:gridClick={e => blocklyWorkspace.setMoveCoords(e.detail.x, e.detail.y)}
    on:gridDrag={e => blocklyWorkspace.setMoveDelta(e.detail.dx, e.detail.dy)}
    on:prevFrame={prevFrame}
    on:nextFrame={nextFrame}
  />
</div>

<style>
  .content { display: flex; flex: 1; overflow: hidden; }
</style>
