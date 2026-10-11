/**
 * Stage: audio extraction + transcription (best-effort, honest fallback).
 * Tries the local `whisper` CLI if installed. If unavailable, the analysis
 * is clearly marked visuals-only — dialogue is never fabricated.
 */
const { execFile } = require("child_process");
const { promisify } = require("util");
const path = require("path");
const fs = require("fs");
const fsp = fs.promises;
const execFileAsync = promisify(execFile);

async function hasWhisper() {
  try { await execFileAsync("whisper", ["--help"]); return true; } catch { return false; }
}

async function hasAudioTrack(videoPath) {
  try {
    const { stdout } = await execFileAsync("ffprobe", [
      "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_name",
      "-of", "csv=p=0", videoPath,
    ]);
    return stdout.trim().length > 0;
  } catch { return false; }
}

async function analyzeAudio(videoPath, workDir) {
  const result = { hasAudio: false, transcribed: false, transcript: null, note: "" };
  result.hasAudio = await hasAudioTrack(videoPath);
  if (!result.hasAudio) { result.note = "No audio track found — analysis covers visuals only."; return result; }
  if (!(await hasWhisper())) {
    result.note = "Audio track present, but no transcription engine is installed on this server — analysis covers visuals only. Install openai-whisper to enable dialogue transcription.";
    return result;
  }
  try {
    const wav = path.join(workDir, "audio-16k.wav");
    await execFileAsync("ffmpeg", ["-y", "-v", "error", "-i", videoPath, "-ar", "16000", "-ac", "1", wav]);
    const outJson = path.join(workDir, "whisper.json");
    await execFileAsync("whisper", [wav, "--model", "base", "--output_format", "json", "--output_dir", workDir, "--fp16", "False"]);
    // whisper names output after input basename
    const produced = path.join(workDir, "audio-16k.json");
    const data = JSON.parse(await fsp.readFile(fs.existsSync(produced) ? produced : outJson, "utf8"));
    result.transcribed = true;
    result.transcript = (data.segments || []).map((s) => ({ start: +s.start.toFixed(2), end: +s.end.toFixed(2), text: s.text.trim() }));
    result.note = `Transcribed ${result.transcript.length} speech segments with whisper (base).`;
  } catch (e) {
    result.note = "Transcription failed (" + e.message + ") — analysis covers visuals only.";
  }
  return result;
}

module.exports = { analyzeAudio };
