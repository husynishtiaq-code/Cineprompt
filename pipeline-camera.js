/**
 * Stage 4: camera-movement estimation from real pixel data.
 * Method: consecutive tiny grayscale frames (2fps, 64x36). For each pair we
 * search integer translations (±10px) and a few zoom scales for the best
 * sum-of-absolute-differences match. Per-scene medians give pan/tilt/zoom
 * trends; variance gives stability (static vs handheld). Low match quality
 * or contradictory motion is reported as "uncertain" — never invented.
 */
const { execFile } = require("child_process");
const { promisify } = require("util");
const execFileAsync = promisify(execFile);

const W = 64, H = 36, FPS = 2;
const SHIFTS = 10;
const SCALES = [0.94, 0.97, 1.0, 1.03, 1.06];

async function grabFrames(videoPath) {
  const { stdout } = await execFileAsync("ffmpeg", [
    "-v", "error", "-i", videoPath, "-vf", `fps=${FPS},scale=${W}:${H}`,
    "-pix_fmt", "gray", "-f", "rawvideo", "-",
  ], { encoding: "buffer", maxBuffer: 256 * 1024 * 1024 });
  const fs = W * H, n = Math.floor(stdout.length / fs), out = [];
  for (let i = 0; i < n; i++) out.push(stdout.subarray(i * fs, (i + 1) * fs));
  return out;
}

function scaleFrame(src, scale) {
  // nearest-neighbor resample back to W x H, centered
  const dst = Buffer.alloc(W * H);
  const sw = W / scale, sh = H / scale, ox = (W - sw) / 2, oy = (H - sh) / 2;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const sx = Math.min(W - 1, Math.max(0, Math.round(ox + x / scale)));
      const sy = Math.min(H - 1, Math.max(0, Math.round(oy + y / scale)));
      dst[y * W + x] = src[sy * W + sx];
    }
  }
  return dst;
}

function sadShift(a, b, dx, dy) {
  let s = 0, n = 0;
  const x0 = Math.max(0, -dx), x1 = Math.min(W, W - dx);
  const y0 = Math.max(0, -dy), y1 = Math.min(H, H - dy);
  for (let y = y0; y < y1; y++) {
    const ra = y * W, rb = (y + dy) * W;
    for (let x = x0; x < x1; x++) { s += Math.abs(a[ra + x] - b[rb + x + dx]); n++; }
  }
  return n ? s / n : Infinity;
}

function bestMatch(a, b) {
  let best = { dx: 0, dy: 0, scale: 1, sad: Infinity };
  for (const sc of SCALES) {
    const bs = sc === 1 ? b : scaleFrame(b, sc);
    for (let dy = -SHIFTS; dy <= SHIFTS; dy++) {
      for (let dx = -SHIFTS; dx <= SHIFTS; dx++) {
        const s = sadShift(a, bs, dx, dy);
        if (s < best.sad) best = { dx, dy, scale: sc, sad: s };
      }
    }
  }
  // confidence: how much better than "no motion" baseline
  const base = sadShift(a, b, 0, 0);
  best.confidence = base > 0 ? Math.max(0, Math.min(1, 1 - best.sad / base)) : 0;
  best.baselineSad = base;
  return best;
}

const median = (arr) => {
  if (!arr.length) return 0;
  const s = [...arr].sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const std = (arr, mean) => {
  if (!arr.length) return 0;
  const mu = mean ?? arr.reduce((a, b) => a + b, 0) / arr.length;
  return Math.sqrt(arr.reduce((a, b) => a + (b - mu) ** 2, 0) / arr.length);
};

async function estimateCamera(videoPath, scenes) {
  const frames = await grabFrames(videoPath);
  if (frames.length < 2) {
    for (const s of scenes) s.camera = { movement: "uncertain", detail: "not enough frames to estimate motion", confidence: 0 };
    return scenes;
  }
  const pairs = [];
  for (let i = 0; i < frames.length - 1; i++) {
    const t = (i + 0.5) / FPS;
    pairs.push({ t, ...bestMatch(frames[i], frames[i + 1]) });
  }
  for (const scene of scenes) {
    const inScene = pairs.filter((p) => p.t >= scene.start && p.t < scene.end);
    if (!inScene.length) {
      scene.camera = { movement: "uncertain", detail: "no motion samples inside this shot", confidence: 0 };
      continue;
    }
    const dxs = inScene.map((p) => p.dx), dys = inScene.map((p) => p.dy);
    const mdx = median(dxs), mdy = median(dys);
    const sdx = std(dxs, mdx), sdy = std(dys, mdy);
    const meanSad = inScene.reduce((a, p) => a + p.sad, 0) / inScene.length;
    const scs = inScene.map((p) => p.scale);
    const slope = (scs[scs.length - 1] - scs[0]) / Math.max(1, scs.length - 1);

    const mag = Math.hypot(mdx, mdy);
    const jitter = Math.hypot(sdx, sdy);
    // vote-based signals (robust to in-scene subject motion)
    const stillVotes = inScene.filter((p) => Math.abs(p.dx) <= 1 && Math.abs(p.dy) <= 1).length;
    const stillFrac = stillVotes / inScene.length;
    const agreeX = inScene.filter((p) => Math.sign(p.dx) === Math.sign(mdx) || Math.abs(p.dx) <= 1).length / inScene.length;
    const agreeY = inScene.filter((p) => Math.sign(p.dy) === Math.sign(mdy) || Math.abs(p.dy) <= 1).length / inScene.length;

    let movement = "uncertain", detail = "", confidence = 0;
    const r2 = (n) => Math.round(n * 100) / 100;

    if (stillFrac >= 0.6 && jitter < 4.5 && Math.abs(slope) < 0.004) {
      movement = "static";
      detail = `locked-off shot — best match is no motion in ${Math.round(stillFrac * 100)}% of sampled pairs`;
      confidence = r2(stillFrac);
    } else if (jitter > 4.5 && stillFrac < 0.4) {
      movement = "handheld";
      detail = `unsteady frame-to-frame motion (jitter σ≈${jitter.toFixed(1)}px)`;
      confidence = r2(Math.min(0.85, 1 - stillFrac));
    } else if (mag >= 0.75 && Math.max(agreeX, agreeY) >= 0.55) {
      const parts = [];
      if (Math.abs(mdx) >= 0.75) parts.push(mdx > 0 ? "pan right" : "pan left");
      if (Math.abs(mdy) >= 0.75) parts.push(mdy > 0 ? "tilt down" : "tilt up");
      if (slope > 0.004) parts.push("zoom in");
      else if (slope < -0.004) parts.push("zoom out");
      if (!parts.length && Math.abs(slope) >= 0.004) parts.push(slope > 0 ? "zoom in" : "zoom out");
      if (!parts.length) {
        movement = "static"; detail = "locked-off shot"; confidence = r2(stillFrac);
      } else {
        movement = parts.join(" + ");
        const speed = mag < 2 ? "slow" : mag < 5 ? "moderate" : "fast";
        detail = `${speed} ${parts.join(", ")} — directional agreement ${Math.round(Math.max(agreeX, agreeY) * 100)}% of pairs`;
        confidence = r2(Math.max(agreeX, agreeY) * 0.9);
      }
    } else {
      movement = "uncertain";
      detail = "mixed or contradictory frame motion — camera path cannot be separated from in-scene movement reliably";
      confidence = r2(0.3 * (1 - stillFrac));
    }
    if (meanSad > 40) confidence = Math.min(confidence, 0.35); // very poor matches cap confidence
    // Distinguish subject-approach vs push-in is not possible from global
    // motion alone; flag it so the vision stage can arbitrate.
    scene.camera = {
      movement, detail, confidence,
      raw: { medianDx: +mdx.toFixed(2), medianDy: +mdy.toFixed(2), jitter: +jitter.toFixed(2), zoomSlope: +slope.toFixed(4), stillFraction: +stillFrac.toFixed(2), meanSad: +meanSad.toFixed(1) },
      needsVisionCheck: movement !== "static" && movement !== "uncertain",
    };
  }
  return scenes;
}

module.exports = { estimateCamera };
