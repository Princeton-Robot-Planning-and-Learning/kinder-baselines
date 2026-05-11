// k6 load test for the deployed Blockly server.
//
// Simulates students running small Blockly programs concurrently. Measures
// end-to-end /run latency, validates that frames come back, and surfaces
// HTTP-level errors (5xx, timeouts, Fly hard-limit rejections).
//
// Run against the default Fly deploy:
//   k6 run loadtest/blockly_load_test.js
//
// Override target URL or peak load:
//   BASE_URL=https://kinder-blockly.fly.dev PEAK_VUS=20 \
//     k6 run loadtest/blockly_load_test.js
//
// Watch Fly logs in another terminal while it runs:
//   fly logs

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend, Rate, Counter } from 'k6/metrics';

const BASE_URL = __ENV.BASE_URL || 'https://kinder-blockly.fly.dev';
const PEAK_VUS = parseInt(__ENV.PEAK_VUS || '15', 10);

const runDuration = new Trend('blockly_run_duration', true);
const resetDuration = new Trend('blockly_reset_duration', true);
const runSuccessful = new Rate('blockly_run_successful');
const framesPerRun = new Trend('blockly_frames_per_run');
const framesReceived = new Counter('blockly_frames_received');

export const options = {
  scenarios: {
    classroom: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '20s', target: Math.max(1, Math.floor(PEAK_VUS / 3)) },
        { duration: '30s', target: Math.max(1, Math.floor(PEAK_VUS / 3)) },
        { duration: '30s', target: PEAK_VUS },
        { duration: '60s', target: PEAK_VUS },
        { duration: '20s', target: 0 },
      ],
      gracefulRampDown: '30s',
    },
  },
  thresholds: {
    blockly_run_duration: ['p(95)<30000'],
    blockly_run_successful: ['rate>0.95'],
    http_req_failed: ['rate<0.05'],
  },
};

// Small program: draw a square with the pen down. Takes a couple of seconds
// of pybullet stepping per run, so it actually exercises the heavy path.
const PROGRAM = {
  blocks: [
    { type: 'set_pen_color', r: 255, g: 0, b: 0 },
    { type: 'pen_down' },
    { type: 'move_base_to_target', x: 0.8, y: 0.0 },
    { type: 'move_base_to_target', x: 0.8, y: 0.8 },
    { type: 'move_base_to_target', x: 0.0, y: 0.8 },
    { type: 'move_base_to_target', x: 0.0, y: 0.0 },
    { type: 'pen_up' },
  ],
};

export function setup() {
  const res = http.get(`${BASE_URL}/healthz`);
  if (res.status !== 200) {
    throw new Error(`healthz returned ${res.status}: ${res.body}`);
  }
  console.log(`target: ${BASE_URL}, peak VUs: ${PEAK_VUS}`);
  return { baseUrl: BASE_URL };
}

export default function (data) {
  // GET / once per iteration to ensure this VU has a kb_session cookie.
  // k6 cookie jars are per-VU and persist across iterations, so we are
  // effectively one student doing many runs in a session.
  const indexRes = http.get(`${data.baseUrl}/`);
  check(indexRes, { 'index 200': (r) => r.status === 200 });

  const resetStart = Date.now();
  const resetRes = http.post(
    `${data.baseUrl}/reset`,
    JSON.stringify({ seed: 0, paint_buckets: [] }),
    { headers: { 'Content-Type': 'application/json' } },
  );
  resetDuration.add(Date.now() - resetStart);
  check(resetRes, { 'reset 200': (r) => r.status === 200 });

  const runStart = Date.now();
  const runRes = http.post(
    `${data.baseUrl}/run`,
    JSON.stringify({ program: PROGRAM, seed: 0, paint_buckets: [] }),
    {
      headers: { 'Content-Type': 'application/json' },
      timeout: '180s',
    },
  );
  runDuration.add(Date.now() - runStart);

  if (runRes.status !== 200) {
    runSuccessful.add(false);
    console.log(`run failed status=${runRes.status} body=${(runRes.body || '').substring(0, 200)}`);
  } else {
    const lines = (runRes.body || '').trim().split('\n');
    let doneMsg = null;
    let frames = 0;
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const msg = JSON.parse(line);
        if (msg.type === 'frame') frames++;
        else if (msg.type === 'done') doneMsg = msg;
      } catch (e) {
        console.log(`bad ndjson line: ${line.substring(0, 100)}`);
      }
    }
    framesReceived.add(frames);
    framesPerRun.add(frames);

    const ok = doneMsg !== null && !doneMsg.error && !doneMsg.infinite_loop;
    runSuccessful.add(ok);
    if (!ok) {
      console.log(
        `run done with error: error=${doneMsg && doneMsg.error} loop=${doneMsg && doneMsg.infinite_loop}`,
      );
    }
  }

  // Think time between runs — students do not click Run continuously.
  sleep(2 + Math.random() * 3);
}
