/**
 * Stage 1-2: video metadata, scene-boundary detection, adaptive frame extraction.
 * All measurements are real (ffprobe + ffmpeg); nothing is simulated.
 */
const { execFile } = require("child_process");
const { promisify } = require("util");
const path = require("path");
const fs = require("fs");
const fsp = fs.promises;

const execFileAsync = promisify(execFile);
function run(cmd, args, opts = {}) {
  return execFileAsync(cmd, args, { maxBuffer: 64 * 1024 * 1024, ...opts });
}

async function getMetadata(videoPath) {
  const { stdout } = await run("ffprobe", [
    "-v", "error",
    "-show_entries", "stream=width,height,avg_frame_rate,duration,codec_name",
    "-show_entries", "format=duration,size",
    "-of", "json", videoPath,
  ]);
  const data = JSON.parse(stdout);
  const v = (data.streams || []).find((s) => s.width) || {};
  const duration = parseFloat(data.format?.duration || v.duration || 0);
  let fps = 0;
  if (v.avg_frame_rate?.includes("/")) {
    const [n, d] = v.avg_frame_rate.split("/").map(Number);
    if (d) fps = n / d;
  }
  const w = v.width || 0, h = v.height || 0;
  let aspect = "16:9";
  if (w && h) {
    const r = w / h;
    aspect = r > 1.7 ? "16:9" : r > 1.3 ? "4:3" : r > 0.8 ? "1:1" : r > 0.6 ? "4:5" : "9:16";
  }
  return { duration: round2(duration), fps: round2(fps), width: w, height: h, aspect, sizeBytes: parseInt(data.format?.size || 0, 10) };
}
const round2 = (n) => Math.round(n * 100) / 100;

/** Detect shot boundaries via ffmpeg scene filter. Returns [{time, score}]. */
async function detectScenes(videoPath, duration, threshold = 0.35) {
  const { stderr } = await run("ffmpeg", [
    "-v", "info", "-i", videoPath,
    "-filter:v", `select='gt(scene,${threshold})',showinfo`,
    "-f", "null", "-",
  ]).catch((e) => ({ stderr: e.stderr || "" }));
  const times = [];
  const re = /pts_time:([\d.]+)/g;
  let m;
  while ((m = re.exec(stderr))) times.push(parseFloat(m[1]));
  // Deduplicate near-identical detections, always include t=0
  const bounds = [0];
  for (const t of times) {
    if (t - bounds[bounds.length - 1] > 0.4) bounds.push(round2(t));
  }
  if (duration - bounds[bounds.length - 1] > 0.4) bounds.push(round2(duration));
  return bounds;
}

/** Split bounds into scene segments, honoring a fixed interval override. */
function buildScenes(bounds, duration, fixedIntervalSec = 0) {
  let segs = [];
  if (fixedIntervalSec > 0) {
    for (let s = 0; s < duration; s += fixedIntervalSec) {
      segs.push({ start: round2(s), end: round2(Math.min(s + fixedIntervalSec, duration)) });
    }
  } else {
    for (let i = 0; i < bounds.length - 1; i++) {
      segs.push({ start: bounds[i], end: bounds[i + 1] });
    }
  }
  // Drop degenerate slivers, merge into neighbor
  segs = segs.filter((s) => s.end - s.start > 0.25);
  return segs.map((s, i) => ({ index: i + 1, ...s, duration: round2(s.end - s.start) }));
}

/** Extract representative frames per scene. Fractions depend on analysis mode. */
async function extractSceneFrames(videoPath, scenes, framesDir, mode) {
  const fractions = mode === "quick" ? [0.5] : mode === "deep" ? [0.05, 0.3, 0.5, 0.7, 0.95] : [0.1, 0.5, 0.9];
  await fsp.mkdir(framesDir, { recursive: true });
  const jobs = [];
  for (const scene of scenes) {
    scene.frames = [];
    for (let fi = 0; fi < fractions.length; fi++) {
      const t = round2(scene.start + scene.duration * fractions[fi]);
      const out = path.join(framesDir, `scene${scene.index}-f${fi}.jpg`);
      scene.frames.push({ time: t, path: out, file: path.basename(out) });
      jobs.push(run("ffmpeg", ["-y", "-v", "error", "-ss", String(t), "-i", videoPath, "-frames:v", "1", "-q:v", "3", out]));
    }
  }
  await Promise.all(jobs);
  return scenes;
}

/** Boundary strength: normalized mean abs diff between small frames across a cut. */
async function boundaryStrength(videoPath, scenes) {
  // Pull tiny grayscale frames at 4fps for cheap differencing
  const W = 48, H = 27;
  const { stdout } = await run("ffmpeg", [
    "-v", "error", "-i", videoPath, "-vf", `fps=4,scale=${W}:${H}`,
    "-pix_fmt", "gray", "-f", "rawvideo", "-",
  ], { encoding: "buffer", maxBuffer: 256 * 1024 * 1024 });
  const buf = stdout;
  const frameSize = W * H;
  const n = Math.floor(buf.length / frameSize);
  const frames = [];
  for (let i = 0; i < n; i++) frames.push(buf.subarray(i * frameSize, (i + 1) * frameSize));
  const fps = 4;
  const diffAt = (t) => {
    const i = Math.max(0, Math.min(n - 2, Math.floor(t * fps)));
    const a = frames[i], b = frames[i + 1];
    let s = 0;
    for (let k = 0; k < frameSize; k++) s += Math.abs(a[k] - b[k]);
    return s / frameSize / 255; // 0..1
  };
  for (const scene of scenes) {
    if (scene.index === 1) { scene.boundaryConfidence = 1; continue; }
    scene.boundaryConfidence = round2(Math.min(1, diffAt(scene.start) * 4));
  }
  return scenes;
}

module.exports = { getMetadata, detectScenes, buildScenes, extractSceneFrames, boundaryStrength, round2 };
