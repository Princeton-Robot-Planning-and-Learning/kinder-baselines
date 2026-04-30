const WORLD_MIN = -2.0;
const WORLD_MAX =  2.0;

// Camera yaw=90: screen right = +Y, screen up = +X.
function worldToCanvas(wx, wy, cw, ch) {
  const range = WORLD_MAX - WORLD_MIN;
  const px = (wy - WORLD_MIN) / range * cw;
  const py = (1 - (wx - WORLD_MIN) / range) * ch;
  return [px, py];
}

export function drawCanvasTrail(canvas, trail) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const range = WORLD_MAX - WORLD_MIN;

  // Dark purple background.
  ctx.fillStyle = '#06020f';
  ctx.fillRect(0, 0, w, h);

  // Grid — subtle purple.
  ctx.strokeStyle = 'rgba(124, 58, 237, 0.15)';
  ctx.lineWidth = 1;
  const step = 0.5;
  for (let g = WORLD_MIN; g <= WORLD_MAX; g += step) {
    const frac = (g - WORLD_MIN) / range;
    const gx = frac * w;
    ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
    const gy = (1 - frac) * h;
    ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
  }

  // Axes — brighter purple.
  ctx.strokeStyle = 'rgba(145, 96, 238, 0.35)';
  ctx.lineWidth = 1.5;
  const origin = worldToCanvas(0, 0, w, h);
  ctx.beginPath(); ctx.moveTo(origin[0], 0); ctx.lineTo(origin[0], h); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, origin[1]); ctx.lineTo(w, origin[1]); ctx.stroke();

  // Numeric labels.
  ctx.fillStyle = 'rgba(184, 131, 250, 0.4)';
  ctx.font = '20px "Silkscreen"';

  // Y-axis numbers (along bottom, horizontal axis on screen).
  ctx.textAlign = 'center';
  for (let ly = WORLD_MIN; ly <= WORLD_MAX; ly += 1) {
    const lp = worldToCanvas(0, ly, w, h);
    ctx.fillText(ly.toFixed(0), lp[0], origin[1] + 11);
  }
  // X-axis numbers (along left, vertical axis on screen).
  ctx.textAlign = 'right';
  for (let lx = WORLD_MIN; lx <= WORLD_MAX; lx += 1) {
    const lp2 = worldToCanvas(lx, 0, w, h);
    ctx.fillText(lx.toFixed(0), origin[0] - 4, lp2[1] + 3);
  }

  // Axis labels — "X" and "Y" with arrows.
  ctx.fillStyle = 'rgba(184, 131, 250, 0.6)';
  ctx.font = '22px "Silkscreen"';
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
  for (let i = 0; i < trail.length; i++) {
    const seg = trail[i];
    const p1 = worldToCanvas(seg.x1, seg.y1, w, h);
    const p2 = worldToCanvas(seg.x2, seg.y2, w, h);
    ctx.strokeStyle = 'rgb(' + seg.r + ',' + seg.g + ',' + seg.b + ')';
    ctx.beginPath(); ctx.moveTo(p1[0], p1[1]); ctx.lineTo(p2[0], p2[1]); ctx.stroke();
  }

  // Trail solid pass.
  ctx.globalAlpha = 1.0;
  ctx.lineWidth = 2.5;
  for (let j = 0; j < trail.length; j++) {
    const s = trail[j];
    const q1 = worldToCanvas(s.x1, s.y1, w, h);
    const q2 = worldToCanvas(s.x2, s.y2, w, h);
    ctx.strokeStyle = 'rgb(' + s.r + ',' + s.g + ',' + s.b + ')';
    ctx.beginPath(); ctx.moveTo(q1[0], q1[1]); ctx.lineTo(q2[0], q2[1]); ctx.stroke();
  }

  // Start dot.
  const first = trail[0];
  const start = worldToCanvas(first.x1, first.y1, w, h);
  ctx.fillStyle = 'rgb(' + first.r + ',' + first.g + ',' + first.b + ')';
  ctx.shadowColor = 'rgb(' + first.r + ',' + first.g + ',' + first.b + ')';
  ctx.shadowBlur = 14;
  ctx.beginPath();
  ctx.arc(start[0], start[1], 5, 0, 2 * Math.PI);
  ctx.fill();

  // End dot with glow.
  const last = trail[trail.length - 1];
  const end = worldToCanvas(last.x2, last.y2, w, h);
  ctx.fillStyle = 'rgb(' + last.r + ',' + last.g + ',' + last.b + ')';
  ctx.shadowColor = 'rgb(' + last.r + ',' + last.g + ',' + last.b + ')';
  ctx.shadowBlur = 18;
  ctx.beginPath();
  ctx.arc(end[0], end[1], 7, 0, 2 * Math.PI);
  ctx.fill();
  ctx.shadowBlur = 0;
}
