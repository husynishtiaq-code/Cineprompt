/**
 * CinePrompt Extractor — backend.
 * Real pipeline: ffprobe metadata → scene detection → adaptive frame
 * extraction → pixel-based camera-motion estimation → audio (whisper if
 * installed) → pluggable vision model → per-scene prompt generation.
 * Secrets stay in server env; the browser never sees API keys.
 */
require("dotenv").config();
const express = require("express");
const multer = require("multer");
const cors = require("cors");
const path = require("path");
const fs = require("fs");
const fsp = fs.promises;
const crypto = require("crypto");

const { getMetadata, detectScenes, buildScenes, extractSceneFrames, boundaryStrength } = require("./pipeline-analyze");
const { estimateCamera } = require("./pipeline-camera");
const { analyzeAudio } = require("./pipeline-audio");
const { analyzeSceneVision } = require("./pipeline-vision");
const { buildScenePrompt, buildCharacterRegistry } = require("./pipeline-prompts");

const PORT = process.env.PORT || 3001;
const ROOT = __dirname;
const UPLOAD_DIR = path.join(ROOT, "uploads");
const WORK_DIR = path.join(ROOT, "work");
const JOB_DIR = path.join(ROOT, "jobs");
for (const d of [UPLOAD_DIR, WORK_DIR, JOB_DIR]) fs.mkdirSync(d, { recursive: true });

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));
app.get("/", (req, res) => res.sendFile(path.join(ROOT, "index.html")));
app.use("/work", express.static(WORK_DIR)); // frame thumbnails

const upload = multer({
  dest: UPLOAD_DIR,
  limits: { fileSize: 500 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const ok = ["video/mp4", "video/quicktime", "video/x-m4v", "video/webm"].includes(file.mimetype)
      || /\.(mp4|mov|m4v|webm)$/i.test(file.originalname);
    cb(ok ? null : new Error("Unsupported video type. Use MP4 or MOV."), ok);
  },
});

// ---------------- job store ----------------
async function saveJob(job) { await fsp.writeFile(path.join(JOB_DIR, `${job.id}.json`), JSON.stringify(job)); }
async function loadJob(id) {
  try { return JSON.parse(await fsp.readFile(path.join(JOB_DIR, `${id}.json`), "utf8")); }
  catch { return null; }
}
function publicJob(j) {
  const { visionCache, ...rest } = j;
  return rest;
}

// ---------------- routes ----------------
app.get("/api/health", (req, res) => {
  res.json({
    ok: true,
    visionProvider: (process.env.VISION_PROVIDER || "none").toLowerCase(),
    visionConfigured:
      (process.env.VISION_PROVIDER === "openai" && !!process.env.OPENAI_API_KEY) ||
      (process.env.VISION_PROVIDER === "gemini" && !!process.env.GEMINI_API_KEY),
    retentionHours: parseInt(process.env.RETENTION_HOURS || "24", 10),
  });
});

app.post("/api/upload", (req, res) => {
  upload.single("video")(req, res, async (err) => {
    if (err) return res.status(400).json({ error: err.message });
    if (!req.file) return res.status(400).json({ error: "No video file received." });
    const uploadId = crypto.randomUUID();
    const ext = path.extname(req.file.originalname) || ".mp4";
    const dest = path.join(UPLOAD_DIR, uploadId + ext);
    await fsp.rename(req.file.path, dest);
    try {
      const meta = await getMetadata(dest);
      if (meta.duration > 900) {
        await fsp.unlink(dest).catch(() => {});
        return res.status(400).json({ error: "Video exceeds the 15-minute processing limit." });
      }
      res.json({ uploadId, fileName: req.file.originalname, meta, videoUrl: `/api/video/${uploadId}` });
    } catch (e) {
      res.status(400).json({ error: "Could not read video file: " + e.message });
    }
  });
});

app.get("/api/video/:id", async (req, res) => {
  const files = await fsp.readdir(UPLOAD_DIR).catch(() => []);
  const f = files.find((x) => x.startsWith(req.params.id));
  if (!f) return res.status(404).end();
  res.sendFile(path.join(UPLOAD_DIR, f));
});

app.post("/api/analyze", async (req, res) => {
  const { uploadId, mode = "standard", language = "en", target = "universal", segInterval = 0 } = req.body || {};
  if (!uploadId) return res.status(400).json({ error: "Missing uploadId." });
  const files = await fsp.readdir(UPLOAD_DIR).catch(() => []);
  const f = files.find((x) => x.startsWith(uploadId));
  if (!f) return res.status(404).json({ error: "Uploaded video not found." });

  const jobId = crypto.randomUUID();
  const job = {
    id: jobId, status: "running", stage: "starting", progress: 2,
    createdAt: new Date().toISOString(),
    options: { mode, language, target, segInterval: Number(segInterval) || 0 },
    uploadId, videoFile: f, result: null, error: null,
  };
  await saveJob(job);
  runPipeline(jobId); // async, polled via GET
  res.json({ jobId });
});

app.get("/api/jobs/:id", async (req, res) => {
  const job = await loadJob(req.params.id);
  if (!job) return res.status(404).json({ error: "Job not found." });
  res.json(publicJob(job));
});

app.get("/api/jobs/:id/export", async (req, res) => {
  const job = await loadJob(req.params.id);
  if (!job?.result) return res.status(404).json({ error: "No result yet." });
  const format = (req.query.format || "txt").toLowerCase();
  const r = job.result;
  if (format === "json") {
    res.setHeader("Content-Disposition", `attachment; filename="cineprompt-${job.id.slice(0, 8)}.json"`);
    return res.json(r);
  }
  // txt
  const lines = [
    `CINEPROMPT EXTRACTOR — ${r.meta.fileName || "video"}`,
    `Duration ${r.meta.duration}s · ${r.meta.width}x${r.meta.height} · ${r.scenes.length} scenes`,
    `Target: ${job.options.target} · Language: ${job.options.language} · Mode: ${job.options.mode}`,
    `Generated ${r.generatedAt}`,
    "",
  ];
  for (const p of r.prompts) lines.push(p.prompt, "", "─".repeat(60), "");
  res.setHeader("Content-Type", "text/plain; charset=utf-8");
  res.setHeader("Content-Disposition", `attachment; filename="cineprompt-${job.id.slice(0, 8)}.txt"`);
  res.send(lines.join("\n"));
});

// Regenerate a single scene's prompt (with optional user correction), no full re-analysis
app.post("/api/jobs/:id/regenerate", async (req, res) => {
  const job = await loadJob(req.params.id);
  if (!job?.result) return res.status(404).json({ error: "No result yet." });
  const { sceneIndex, correction = "" } = req.body || {};
  const entry = job.visionCache?.find((v) => v.scene.index === sceneIndex);
  if (!entry) return res.status(404).json({ error: "Scene not found." });
  try {
    const provider = (process.env.VISION_PROVIDER || "none").toLowerCase();
    if (provider === "none") return res.status(400).json({ error: "Vision provider is not configured — cannot regenerate visual analysis." });
    if (correction) entry.scene.userCorrection = correction;
    const vision = await analyzeSceneVision(entry.scene, job.options.language, entry.scene.camera, job.result.audio.transcribed);
    entry.vision = vision;
    const rebuilt = buildScenePrompt(entry.scene, vision, job.result.audio, job.options.target);
    const i = job.result.prompts.findIndex((p) => p.scene === sceneIndex);
    job.result.prompts[i] = rebuilt;
    job.result.characters = buildCharacterRegistry(job.visionCache.map((v) => ({ scene: v.scene, vision: v.vision })));
    await saveJob(job);
    res.json({ prompt: rebuilt });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Manual merge/split: replace scene list with user-edited segments, rebuild prompts
app.post("/api/jobs/:id/scenes", async (req, res) => {
  const job = await loadJob(req.params.id);
  if (!job?.result) return res.status(404).json({ error: "No result yet." });
  const { segments } = req.body || {}; // [{start,end}]
  if (!Array.isArray(segments) || !segments.length) return res.status(400).json({ error: "Provide segments [{start,end}]." });
  try {
    const framesDir = path.join(WORK_DIR, job.id, "frames");
    let scenes = segments.map((s, i) => ({ index: i + 1, start: +s.start, end: +s.end, duration: Math.round((+s.end - +s.start) * 100) / 100 }));
    scenes = await extractSceneFrames(path.join(UPLOAD_DIR, job.videoFile), scenes, framesDir, job.options.mode);
    scenes = await boundaryStrength(path.join(UPLOAD_DIR, job.videoFile), scenes);
    scenes = await estimateCamera(path.join(UPLOAD_DIR, job.videoFile), scenes);
    const visionCache = [];
    for (const scene of scenes) {
      const old = job.visionCache?.find((v) => Math.abs(v.scene.start - scene.start) < 0.6 && Math.abs(v.scene.end - scene.end) < 0.6);
      const vision = old?.vision || await analyzeSceneVision(scene, job.options.language, scene.camera, job.result.audio.transcribed);
      visionCache.push({ scene, vision });
    }
    job.visionCache = visionCache;
    job.result.scenes = scenes.map((s) => ({ index: s.index, start: s.start, end: s.end, duration: s.duration, boundaryConfidence: s.boundaryConfidence, camera: s.camera, frames: s.frames.map((f) => f.file) }));
    job.result.prompts = visionCache.map(({ scene, vision }) => buildScenePrompt(scene, vision, job.result.audio, job.options.target));
    job.result.characters = buildCharacterRegistry(visionCache.map((v) => ({ scene: v.scene, vision: v.vision })));
    await saveJob(job);
    res.json(publicJob(job));
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ---------------- pipeline ----------------
async function runPipeline(jobId) {
  const job = await loadJob(jobId);
  const set = async (stage, progress) => { job.stage = stage; job.progress = progress; await saveJob(job); };
  try {
    const videoPath = path.join(UPLOAD_DIR, job.videoFile);
    const workDir = path.join(WORK_DIR, jobId);
    const framesDir = path.join(workDir, "frames");
    await fsp.mkdir(framesDir, { recursive: true });
    const { mode, language, target, segInterval } = job.options;

    await set("Reading video metadata", 6);
    const meta = await getMetadata(videoPath);
    const origName = job.videoFile;

    await set("Detecting scene boundaries", 14);
    const bounds = await detectScenes(videoPath, meta.duration);
    let scenes = buildScenes(bounds, meta.duration, segInterval);

    await set(`Extracting keyframes (${mode} mode)`, 26);
    scenes = await extractSceneFrames(videoPath, scenes, framesDir, mode);

    await set("Measuring shot-boundary strength", 36);
    scenes = await boundaryStrength(videoPath, scenes);

    await set("Estimating camera motion", 48);
    scenes = await estimateCamera(videoPath, scenes);

    await set("Checking audio track", 56);
    const audio = await analyzeAudio(videoPath, workDir);

    const provider = (process.env.VISION_PROVIDER || "none").toLowerCase();
    const visionCache = [];
    if (provider === "none") {
      await set("Vision provider not configured — assembling measured data only", 70);
      for (const scene of scenes) visionCache.push({ scene, vision: null });
    } else {
      let i = 0;
      for (const scene of scenes) {
        i++;
        await set(`Analyzing shot ${i}/${scenes.length} with vision model`, 56 + Math.round((i / scenes.length) * 30));
        const vision = await analyzeSceneVision(scene, language, scene.camera, audio.transcribed);
        visionCache.push({ scene, vision });
      }
    }

    await set("Generating prompts", 94);
    const prompts = visionCache.map(({ scene, vision }) => buildScenePrompt(scene, vision, audio, target));
    const characters = buildCharacterRegistry(visionCache.map((v) => ({ scene: v.scene, vision: v.vision })));

    job.result = {
      generatedAt: new Date().toISOString(),
      meta: { ...meta, fileName: origName },
      visionAvailable: provider !== "none",
      visionProvider: provider,
      audio: { hasAudio: audio.hasAudio, transcribed: audio.transcribed, note: audio.note, transcript: audio.transcript },
      cameraSummary: summarizeCamera(scenes),
      scenes: scenes.map((s) => ({
        index: s.index, start: s.start, end: s.end, duration: s.duration,
        boundaryConfidence: s.boundaryConfidence, camera: s.camera,
        frames: s.frames.map((f) => ({ time: f.time, url: `/work/${jobId}/frames/${f.file}` })),
      })),
      characters,
      prompts,
    };
    job.visionCache = visionCache.map(({ scene, vision }) => ({
      scene: { index: scene.index, start: scene.start, end: scene.end, duration: scene.duration, camera: scene.camera, frames: scene.frames },
      vision,
    }));
    job.status = "done"; job.progress = 100; job.stage = "complete";
    await saveJob(job);
  } catch (e) {
    job.status = "failed"; job.error = e.message; job.stage = "failed";
    await saveJob(job);
  }
}

function summarizeCamera(scenes) {
  const counts = {};
  for (const s of scenes) {
    const m = s.camera?.movement || "uncertain";
    counts[m] = (counts[m] || 0) + 1;
  }
  return Object.entries(counts).map(([movement, count]) => ({ movement, shots: count }));
}

// Retention sweep on startup
(async () => {
  const hours = parseInt(process.env.RETENTION_HOURS || "24", 10);
  if (!hours) return;
  const cutoff = Date.now() - hours * 3600 * 1000;
  for (const dir of [UPLOAD_DIR, WORK_DIR]) {
    for (const f of await fsp.readdir(dir).catch(() => [])) {
      const p = path.join(dir, f);
      try {
        const st = await fsp.stat(p);
        if (st.mtimeMs < cutoff) await fsp.rm(p, { recursive: true, force: true });
      } catch { }
    }
  }
})();

app.listen(PORT, () => {
  console.log(`CinePrompt Extractor on http://localhost:${PORT} (vision: ${process.env.VISION_PROVIDER || "none"})`);
});
