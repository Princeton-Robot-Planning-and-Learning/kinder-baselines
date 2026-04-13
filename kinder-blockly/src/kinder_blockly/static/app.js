/* KinDER Blockly — retro purple edition */

/* ── Retro Blockly theme ───────────────────────────────────────── */

var retroTheme = Blockly.Theme.defineTheme('retro', {
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
    fontStyle: { family: "'Press Start 2P', monospace", size: 9 },
});

var workspace = Blockly.inject('blockly-div', {
    toolbox: document.getElementById('toolbox'),
    theme: retroTheme,
    trashcan: true,
    scrollbars: true,
    renderer: 'zelos',
    grid: { spacing: 25, length: 3, colour: '#1a0a3d', snap: true },
});

/* ── State ─────────────────────────────────────────────────────── */

var currentChallenge = null;
var lastStudentTrail = [];

/* ── Tamagotchi ────────────────────────────────────────────────── */

var tamaTimeout = null;

// var TAMA_IDLE_TIPS = [
//     "I YEARN to draw... please, give me blocks!",
//     "TIP: PEN UP lets me glide without leaving a mark. Stealth mode.",
//     "TIP: You can change colours mid-stroke! I contain multitudes.",
//     "TIP: I start at (0, 0). Humble beginnings for a great artist.",
//     "Every masterpiece begins with a single block...",
//     "HINT: The Target canvas haunts my dreams. I MUST replicate it.",
//     "HINT: Try simple shapes first! Even Picasso started somewhere.",
//     "TIP: X and Y go from -2 to 2. That's my whole world. It's enough.",
//     "Click me! I'm lonely and full of wisdom!",
//     "TIP: X goes up, Y goes right. I didn't make the rules.",
//     "I was BORN to draw. Tidying was just my day job.",
//     "A robot without a pen is like a bird without wings. Tragic.",
//     "They said I could be anything... so I became an ARTIST.",
//     "My circuits tingle when you drag blocks. Keep going!",
//     "One does not simply free draw. One EXPRESSES oneself.",
//     "HINT: Check the Target canvas for your goal!",
//     "TIP: Click me any time for a hint...",
// ];
var TAMA_IDLE_TIPS = ["I was BORN to draw. Tidying was just my day job.",];
function tamaSay(msg, duration) {
    var bubble = document.getElementById('tama-bubble');
    bubble.textContent = msg;
    bubble.classList.add('visible');
    if (tamaTimeout) clearTimeout(tamaTimeout);
    tamaTimeout = setTimeout(function() {
        bubble.classList.remove('visible');
    }, duration || 5000);
}

function tamaPoke() {
    var tip = TAMA_IDLE_TIPS[Math.floor(Math.random() * TAMA_IDLE_TIPS.length)];
    tamaSay(tip, 4000);
}

setTimeout(function() {
    tamaSay("Greetings, young artist! I am TidyBot, and I LIVE to draw. Pick a challenge... or let me free draw. Please.", 6000);
}, 1500);

/* ── Initial frame ─────────────────────────────────────────────── */

function loadInitialFrame() {
    document.getElementById('frame-info').textContent = 'LOADING WORLD...';
    fetch('/reset', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({seed: 0}),
    })
    .then(function(resp) { return resp.json(); })
    .then(function(data) {
        if (data.frame) {
            document.getElementById('frame-display').src =
                'data:image/jpeg;base64,' + data.frame;
            document.getElementById('frame-info').textContent = 'READY!';
        }
    })
    .catch(function(err) {
        document.getElementById('frame-info').textContent = 'LOAD FAILED: ' + err;
    });
}

loadInitialFrame();

/* ── Challenge loading ─────────────────────────────────────────── */

function loadChallenges() {
    fetch('/challenges')
    .then(function(r) { return r.json(); })
    .then(function(data) {
        var sel = document.getElementById('challenge-select');
        (data.challenges || []).forEach(function(c) {
            var opt = document.createElement('option');
            opt.value = c.id;
            var stars = c.difficulty === 'easy' ? '*' : c.difficulty === 'medium' ? '**' : '***';
            opt.textContent = '[' + stars + '] ' + c.name.toUpperCase();
            opt.dataset.hint = c.hint || '';
            opt.dataset.description = c.description || '';
            sel.appendChild(opt);
        });
    })
    .catch(function() {});
}

loadChallenges();

function onChallengeChange() {
    var sel = document.getElementById('challenge-select');
    var id = sel.value;
    var hint = document.getElementById('challenge-hint');
    var scoreBox = document.getElementById('score-box');
    scoreBox.className = 'retro-border';
    scoreBox.style.display = 'none';
    scoreBox.textContent = '';

    if (!id) {
        currentChallenge = null;
        hint.textContent = '';
        drawCanvasTrail('target-canvas', []);
        tamaSay("FREE DRAW! No rules, no limits! Just me and the canvas. *chef's kiss*", 4000);
        return;
    }

    hint.textContent = 'LOADING...';
    fetch('/challenges/' + id)
    .then(function(r) { return r.json(); })
    .then(function(c) {
        currentChallenge = c;
        hint.textContent = c.description || '';
        drawCanvasTrail('target-canvas', c.target_trail || []);
        tamaSay(c.hint || c.description || "Good luck!", 5000);
    })
    .catch(function() { hint.textContent = 'FAILED TO LOAD.'; });
}

/* ── Extract program ───────────────────────────────────────────── */

function extractProgram() {
    var blocks = [];
    var topBlocks = workspace.getTopBlocks(true);
    for (var i = 0; i < topBlocks.length; i++) {
        var block = topBlocks[i];
        while (block) {
            var entry = {type: block.type};
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
    return {blocks: blocks};
}

/* ── Run program ───────────────────────────────────────────────── */

function runProgram() {
    var program = extractProgram();
    if (program.blocks.length === 0) {
        document.getElementById('status').textContent = 'NO BLOCKS!';
        tamaSay("Drag some blocks first!", 3000);
        return;
    }

    var btn = document.getElementById('run-btn');
    btn.disabled = true;
    document.getElementById('status').textContent = 'RUNNING...';
    document.getElementById('frame-info').textContent = '';
    var scoreBox = document.getElementById('score-box');
    scoreBox.className = 'retro-border';
    scoreBox.style.display = 'none';

    tamaSay("Let's go! Executing program...", 3000);

    fetch('/run', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({program: program, seed: 0}),
    })
    .then(function(resp) { return resp.json(); })
    .then(function(data) {
        btn.disabled = false;
        if (data.error) {
            document.getElementById('status').textContent = 'ERROR!';
            tamaSay("Oops! " + data.error, 5000);
        } else {
            document.getElementById('status').textContent = 'DONE!';
        }
        playFrames(data.frames || []);
        lastStudentTrail = data.trail || [];
        drawCanvasTrail('trail-canvas', lastStudentTrail);

        if (currentChallenge && lastStudentTrail.length > 0) {
            requestScore(currentChallenge.id, lastStudentTrail);
        } else if (!currentChallenge) {
            tamaSay("Nice drawing! Pick a challenge to get scored!", 4000);
        }
    })
    .catch(function() {
        btn.disabled = false;
        document.getElementById('status').textContent = 'ERROR!';
        tamaSay("Connection error! Is the server running?", 4000);
    });
}

/* ── Scoring ───────────────────────────────────────────────────── */

function requestScore(challengeId, studentTrail) {
    fetch('/score', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({challenge_id: challengeId, student_trail: studentTrail}),
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
        showScore(data.score, data.breakdown);
    })
    .catch(function(err) { console.error('Score request failed:', err); });
}

function showScore(score, breakdown) {
    var box = document.getElementById('score-box');
    var cls = score >= 70 ? 'good' : score >= 40 ? 'ok' : 'poor';
    box.className = 'retro-border ' + cls;
    box.style.display = '';
    box.innerHTML = 'SCORE: ' + score + ' / 100'
        + '<div id="score-detail">'
        + 'Coverage ' + breakdown.coverage + '%'
        + ' // Precision ' + breakdown.precision + '%'
        + ' // Colour ' + breakdown.color + '%'
        + '</div>';

    if (score >= 90) {
        tamaSay("PERFECT! You're a robot artist!", 5000);
    } else if (score >= 70) {
        tamaSay("Great job! Almost perfect!", 4000);
    } else if (score >= 40) {
        tamaSay("Getting there! Check the target and try again!", 5000);
    } else {
        tamaSay("Keep trying! Compare your drawing to the target.", 5000);
    }
}

/* ── Frame playback ────────────────────────────────────────────── */

function playFrames(frames) {
    if (frames.length === 0) {
        document.getElementById('frame-info').textContent = 'NO FRAMES';
        return;
    }
    var img = document.getElementById('frame-display');
    var idx = 0;
    var info = document.getElementById('frame-info');

    function showFrame() {
        img.src = 'data:image/jpeg;base64,' + frames[idx];
        info.textContent = 'FRAME ' + (idx + 1) + '/' + frames.length;
        idx++;
        if (idx < frames.length) {
            setTimeout(showFrame, 100);
        }
    }
    showFrame();
}

/* ── Canvas drawing ────────────────────────────────────────────── */

var WORLD_MIN = -2.0;
var WORLD_MAX =  2.0;

// Camera yaw=90: screen right = +Y, screen up = +X.
function worldToCanvas(wx, wy, cw, ch) {
    var range = WORLD_MAX - WORLD_MIN;
    var px = (wy - WORLD_MIN) / range * cw;
    var py = (1 - (wx - WORLD_MIN) / range) * ch;
    return [px, py];
}

function drawCanvasTrail(canvasId, trail) {
    var canvas = document.getElementById(canvasId);
    var ctx = canvas.getContext('2d');
    var w = canvas.width;
    var h = canvas.height;
    var range = WORLD_MAX - WORLD_MIN;

    // Dark purple background.
    ctx.fillStyle = '#06020f';
    ctx.fillRect(0, 0, w, h);

    // Grid — subtle purple.
    ctx.strokeStyle = 'rgba(124, 58, 237, 0.15)';
    ctx.lineWidth = 1;
    var step = 0.5;
    for (var g = WORLD_MIN; g <= WORLD_MAX; g += step) {
        var frac = (g - WORLD_MIN) / range;
        var gx = frac * w;
        ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
        var gy = (1 - frac) * h;
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
    }

    // Axes — brighter purple.
    ctx.strokeStyle = 'rgba(145, 96, 238, 0.35)';
    ctx.lineWidth = 1.5;
    var origin = worldToCanvas(0, 0, w, h);
    ctx.beginPath(); ctx.moveTo(origin[0], 0); ctx.lineTo(origin[0], h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, origin[1]); ctx.lineTo(w, origin[1]); ctx.stroke();

    // Numeric labels.
    ctx.fillStyle = 'rgba(184, 131, 250, 0.4)';
    ctx.font = '7px "Press Start 2P"';

    // Y-axis numbers (along bottom, horizontal axis on screen).
    ctx.textAlign = 'center';
    for (var ly = WORLD_MIN; ly <= WORLD_MAX; ly += 1) {
        var lp = worldToCanvas(0, ly, w, h);
        ctx.fillText(ly.toFixed(0), lp[0], origin[1] + 11);
    }
    // X-axis numbers (along left, vertical axis on screen).
    ctx.textAlign = 'right';
    for (var lx = WORLD_MIN; lx <= WORLD_MAX; lx += 1) {
        var lp2 = worldToCanvas(lx, 0, w, h);
        ctx.fillText(lx.toFixed(0), origin[0] - 4, lp2[1] + 3);
    }

    // Axis labels — "X" and "Y" with arrows.
    ctx.fillStyle = 'rgba(184, 131, 250, 0.6)';
    ctx.font = '8px "Press Start 2P"';
    // X label at top of vertical axis (X points up on screen).
    ctx.textAlign = 'left';
    ctx.fillText('X', origin[0] + 4, 12);
    // Small arrow pointing up.
    ctx.beginPath();
    ctx.moveTo(origin[0], 2); ctx.lineTo(origin[0] - 4, 10); ctx.lineTo(origin[0] + 4, 10);
    ctx.closePath(); ctx.fill();

    // Y label at right end of horizontal axis (Y points right on screen).
    ctx.textAlign = 'right';
    ctx.fillText('Y', w - 4, origin[1] - 6);
    // Small arrow pointing right.
    ctx.beginPath();
    ctx.moveTo(w - 2, origin[1]); ctx.lineTo(w - 10, origin[1] - 4); ctx.lineTo(w - 10, origin[1] + 4);
    ctx.closePath(); ctx.fill();

    if (!trail || trail.length === 0) return;

    // Trail glow pass (purple tinted).
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = 6;
    ctx.globalAlpha = 0.2;
    for (var i = 0; i < trail.length; i++) {
        var seg = trail[i];
        var p1 = worldToCanvas(seg.x1, seg.y1, w, h);
        var p2 = worldToCanvas(seg.x2, seg.y2, w, h);
        ctx.strokeStyle = 'rgb(' + seg.r + ',' + seg.g + ',' + seg.b + ')';
        ctx.beginPath(); ctx.moveTo(p1[0], p1[1]); ctx.lineTo(p2[0], p2[1]); ctx.stroke();
    }

    // Trail solid pass.
    ctx.globalAlpha = 1.0;
    ctx.lineWidth = 2.5;
    for (var j = 0; j < trail.length; j++) {
        var s = trail[j];
        var q1 = worldToCanvas(s.x1, s.y1, w, h);
        var q2 = worldToCanvas(s.x2, s.y2, w, h);
        ctx.strokeStyle = 'rgb(' + s.r + ',' + s.g + ',' + s.b + ')';
        ctx.beginPath(); ctx.moveTo(q1[0], q1[1]); ctx.lineTo(q2[0], q2[1]); ctx.stroke();
    }

    // End dot with glow.
    var last = trail[trail.length - 1];
    var end = worldToCanvas(last.x2, last.y2, w, h);
    ctx.fillStyle = 'rgb(' + last.r + ',' + last.g + ',' + last.b + ')';
    ctx.shadowColor = 'rgb(' + last.r + ',' + last.g + ',' + last.b + ')';
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.arc(end[0], end[1], 3, 0, 2 * Math.PI);
    ctx.fill();
    ctx.shadowBlur = 0;
}
