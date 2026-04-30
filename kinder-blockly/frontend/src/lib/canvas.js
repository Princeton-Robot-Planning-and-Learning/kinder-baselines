const WORLD_MIN = -2.0;
const WORLD_MAX =  2.0;

// Camera yaw=90: screen right = +Y, screen up = +X.
export function worldToCanvas(wx, wy, cw, ch) {
  const range = WORLD_MAX - WORLD_MIN;
  const px = (wy - WORLD_MIN) / range * cw;
  const py = (1 - (wx - WORLD_MIN) / range) * ch;
  return [px, py];
}

export function drawVectorDrag(canvas, wx1, wy1, wx2, wy2) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const [px1, py1] = worldToCanvas(wx1, wy1, w, h);
  const [px2, py2] = worldToCanvas(wx2, wy2, w, h);
  const color = '#a78bfa';
  ctx.shadowColor = color;
  ctx.shadowBlur = 12;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;
  // Line
  ctx.beginPath(); ctx.moveTo(px1, py1); ctx.lineTo(px2, py2); ctx.stroke();
  // Start dot
  ctx.beginPath(); ctx.arc(px1, py1, 4, 0, 2 * Math.PI); ctx.fill();
  // Arrowhead
  const angle = Math.atan2(py2 - py1, px2 - px1);
  const aLen = 14;
  const aAngle = Math.PI / 6;
  ctx.beginPath();
  ctx.moveTo(px2, py2);
  ctx.lineTo(px2 - aLen * Math.cos(angle - aAngle), py2 - aLen * Math.sin(angle - aAngle));
  ctx.lineTo(px2 - aLen * Math.cos(angle + aAngle), py2 - aLen * Math.sin(angle + aAngle));
  ctx.closePath(); ctx.fill();
  ctx.shadowBlur = 0;
}

export function drawHoverMarker(canvas, wx, wy) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const [px, py] = worldToCanvas(wx, wy, w, h);
  // Dashed crosshair
  ctx.strokeStyle = 'rgba(167, 139, 250, 0.5)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, h); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(w, py); ctx.stroke();
  ctx.setLineDash([]);
  // Glowing dot
  ctx.shadowColor = '#a78bfa';
  ctx.shadowBlur = 16;
  ctx.fillStyle = '#a78bfa';
  ctx.beginPath(); ctx.arc(px, py, 5, 0, 2 * Math.PI); ctx.fill();
  ctx.shadowBlur = 0;
}

export function canvasToWorld(px, py, cw, ch) {
  const range = WORLD_MAX - WORLD_MIN;
  const wy = px / cw * range + WORLD_MIN;
  const wx = (1 - py / ch) * range + WORLD_MIN;
  return [wx, wy];
}

export function drawPenMarkers(canvas, events) {
  if (!events || events.length === 0) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  // Merge events at the same position: track which types appear at each spot.
  const posMap = new Map();
  for (const ev of events) {
    const key = ev.x.toFixed(3) + ',' + ev.y.toFixed(3);
    if (!posMap.has(key)) posMap.set(key, { x: ev.x, y: ev.y, types: new Set(), r: ev.r, g: ev.g, b: ev.b });
    posMap.get(key).types.add(ev.type);
  }

  for (const { x, y, types, r, g, b } of posMap.values()) {
    const [px, py] = worldToCanvas(-x, y, w, h);
    const color = 'rgb(' + r + ',' + g + ',' + b + ')';
    ctx.strokeStyle = color;
    ctx.fillStyle = 'none';
    ctx.shadowColor = color;
    ctx.shadowBlur = 10;
    ctx.lineWidth = 2;

    if (types.has('down')) {
      // × — pen entering the page
      const s = 5;
      ctx.beginPath();
      ctx.moveTo(px - s, py - s); ctx.lineTo(px + s, py + s);
      ctx.moveTo(px + s, py - s); ctx.lineTo(px - s, py + s);
      ctx.stroke();
    }

    if (types.has('up')) {
      // ○ — pen leaving the page
      ctx.beginPath();
      ctx.arc(px, py, 7, 0, 2 * Math.PI);
      ctx.stroke();
    }

    ctx.shadowBlur = 0;
  }
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

  // Axis labels — "X" horizontal, "Y" vertical.
  ctx.fillStyle = 'rgba(184, 131, 250, 0.6)';
  ctx.font = '22px "Silkscreen"';
  // X label at right end of horizontal axis.
  ctx.textAlign = 'right';
  ctx.fillText('X', w - 4, origin[1] - 6);
  // Small arrow pointing right.
  ctx.beginPath();
  ctx.moveTo(w - 2, origin[1]); ctx.lineTo(w - 10, origin[1] - 4); ctx.lineTo(w - 10, origin[1] + 4);
  ctx.closePath(); ctx.fill();

  // Y label at top of vertical axis.
  ctx.textAlign = 'left';
  ctx.fillText('Y', origin[0] + 4, 12);
  // Small arrow pointing up.
  ctx.beginPath();
  ctx.moveTo(origin[0], 2); ctx.lineTo(origin[0] - 4, 10); ctx.lineTo(origin[0] + 4, 10);
  ctx.closePath(); ctx.fill();

  if (!trail || trail.length === 0) return;

  // Trail glow pass (purple tinted).
  // Trail x values are robot X; negate to convert to UI Y for worldToCanvas.
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.lineWidth = 6;
  ctx.globalAlpha = 0.2;
  for (let i = 0; i < trail.length; i++) {
    const seg = trail[i];
    const p1 = worldToCanvas(-seg.x1, seg.y1, w, h);
    const p2 = worldToCanvas(-seg.x2, seg.y2, w, h);
    ctx.strokeStyle = 'rgb(' + seg.r + ',' + seg.g + ',' + seg.b + ')';
    ctx.beginPath(); ctx.moveTo(p1[0], p1[1]); ctx.lineTo(p2[0], p2[1]); ctx.stroke();
  }

  // Trail solid pass.
  ctx.globalAlpha = 1.0;
  ctx.lineWidth = 2.5;
  for (let j = 0; j < trail.length; j++) {
    const s = trail[j];
    const q1 = worldToCanvas(-s.x1, s.y1, w, h);
    const q2 = worldToCanvas(-s.x2, s.y2, w, h);
    ctx.strokeStyle = 'rgb(' + s.r + ',' + s.g + ',' + s.b + ')';
    ctx.beginPath(); ctx.moveTo(q1[0], q1[1]); ctx.lineTo(q2[0], q2[1]); ctx.stroke();
  }

  ctx.shadowBlur = 0;
}
