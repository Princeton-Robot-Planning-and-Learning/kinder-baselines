/* Blockly workspace setup, program execution, and challenge system. */

var workspace = Blockly.inject('blockly-div', {
    toolbox: document.getElementById('toolbox'),
    trashcan: true,
    scrollbars: true,
});

/* ── State ─────────────────────────────────────────────────────── */

var currentChallenge = null;   // full challenge object (with target_trail)
var lastStudentTrail = [];     // trail from most recent run

/* ── Initial frame ─────────────────────────────────────────────── */

function loadInitialFrame() {
    document.getElementById('frame-info').textContent = 'Loading environment...';
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
            document.getElementById('frame-info').textContent =
                'Drag blocks and click Run';
        }
    })
    .catch(function(err) {
        document.getElementById('frame-info').textContent = 'Failed to load: ' + err;
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
            opt.textContent = c.name + ' (' + c.difficulty + ')';
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
    scoreBox.className = '';
    scoreBox.style.display = 'none';
    scoreBox.textContent = '';

    if (!id) {
        currentChallenge = null;
        hint.textContent = '';
        drawCanvasTrail('target-canvas', []);
        return;
    }

    hint.textContent = 'Loading...';
    fetch('/challenges/' + id)
    .then(function(r) { return r.json(); })
    .then(function(c) {
        currentChallenge = c;
        hint.textContent = c.hint || c.description || '';
        drawCanvasTrail('target-canvas', c.target_trail || []);
    })
    .catch(function() { hint.textContent = 'Failed to load challenge.'; });
}

/* ── Extract program from workspace ────────────────────────────── */

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
        document.getElementById('status').textContent = 'No blocks to run.';
        return;
    }

    var btn = document.getElementById('run-btn');
    btn.disabled = true;
    document.getElementById('status').textContent = 'Running...';
    document.getElementById('frame-info').textContent = '';
    var scoreBox = document.getElementById('score-box');
    scoreBox.className = '';
    scoreBox.style.display = 'none';

    fetch('/run', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({program: program, seed: 0}),
    })
    .then(function(resp) { return resp.json(); })
    .then(function(data) {
        btn.disabled = false;
        if (data.error) {
            document.getElementById('status').textContent = 'Error: ' + data.error;
        } else {
            document.getElementById('status').textContent = 'Done!';
        }
        playFrames(data.frames || []);
        lastStudentTrail = data.trail || [];
        drawCanvasTrail('trail-canvas', lastStudentTrail);

        // Auto-score if a challenge is active.
        if (currentChallenge && lastStudentTrail.length > 0) {
            requestScore(currentChallenge.id, lastStudentTrail);
        }
    })
    .catch(function(err) {
        btn.disabled = false;
        document.getElementById('status').textContent = 'Error: ' + err;
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
    box.className = cls;
    box.style.display = '';  // clear the inline override so the CSS class wins
    box.innerHTML = 'Score: ' + score + ' / 100'
        + '<div id="score-detail">'
        + 'Coverage ' + breakdown.coverage + '%'
        + ' &middot; Precision ' + breakdown.precision + '%'
        + ' &middot; Colour ' + breakdown.color + '%'
        + '</div>';
}

/* ── 3D frame playback ─────────────────────────────────────────── */

function playFrames(frames) {
    if (frames.length === 0) {
        document.getElementById('frame-info').textContent = 'No frames.';
        return;
    }
    var img = document.getElementById('frame-display');
    var idx = 0;
    var info = document.getElementById('frame-info');

    function showFrame() {
        img.src = 'data:image/jpeg;base64,' + frames[idx];
        info.textContent = 'Frame ' + (idx + 1) + ' / ' + frames.length;
        idx++;
        if (idx < frames.length) {
            setTimeout(showFrame, 100);
        }
    }
    showFrame();
}

/* ── Canvas drawing (shared by target + student) ───────────────── */

var WORLD_MIN = -2.0;
var WORLD_MAX =  2.0;

// Rotated 90 deg CW to match the 3-D render camera (yaw = 90):
//   screen right = world +Y,  screen up = world +X.
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

    // Clear.
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);

    // Grid.
    ctx.strokeStyle = '#e8e8e8';
    ctx.lineWidth = 1;
    var range = WORLD_MAX - WORLD_MIN;
    var step = 0.5;
    for (var g = WORLD_MIN; g <= WORLD_MAX; g += step) {
        var frac = (g - WORLD_MIN) / range;
        var gx = frac * w;
        ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
        var gy = (1 - frac) * h;
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
    }

    // Axes.
    ctx.strokeStyle = '#ccc';
    ctx.lineWidth = 1;
    var origin = worldToCanvas(0, 0, w, h);
    ctx.beginPath(); ctx.moveTo(origin[0], 0); ctx.lineTo(origin[0], h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, origin[1]); ctx.lineTo(w, origin[1]); ctx.stroke();

    // Axis labels.
    ctx.fillStyle = '#aaa';
    ctx.font = '9px system-ui';
    ctx.textAlign = 'center';
    for (var ly = WORLD_MIN; ly <= WORLD_MAX; ly += 1) {
        var lp = worldToCanvas(0, ly, w, h);
        ctx.fillText('y=' + ly.toFixed(0), lp[0], origin[1] + 10);
    }
    ctx.textAlign = 'right';
    for (var lx = WORLD_MIN; lx <= WORLD_MAX; lx += 1) {
        var lp2 = worldToCanvas(lx, 0, w, h);
        ctx.fillText('x=' + lx.toFixed(0), origin[0] - 3, lp2[1] + 3);
    }

    // Trail segments.
    if (!trail || trail.length === 0) return;

    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = 2.5;

    for (var i = 0; i < trail.length; i++) {
        var seg = trail[i];
        var p1 = worldToCanvas(seg.x1, seg.y1, w, h);
        var p2 = worldToCanvas(seg.x2, seg.y2, w, h);
        ctx.strokeStyle = 'rgb(' + seg.r + ',' + seg.g + ',' + seg.b + ')';
        ctx.beginPath();
        ctx.moveTo(p1[0], p1[1]);
        ctx.lineTo(p2[0], p2[1]);
        ctx.stroke();
    }

    // Dot at final position.
    var last = trail[trail.length - 1];
    var end = worldToCanvas(last.x2, last.y2, w, h);
    ctx.fillStyle = 'rgb(' + last.r + ',' + last.g + ',' + last.b + ')';
    ctx.beginPath();
    ctx.arc(end[0], end[1], 3, 0, 2 * Math.PI);
    ctx.fill();
}
