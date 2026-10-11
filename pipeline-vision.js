/**
 * Stage 5: visual understanding via a pluggable multimodal model.
 * Providers: "openai" | "gemini" | "none".
 * With "none" this module returns null and the app honestly reports that
 * visual descriptions are unavailable — nothing is simulated.
 */
const { execFile } = require("child_process");
const { promisify } = require("util");
const path = require("path");
const fs = require("fs");
const fsp = fs.promises;
const execFileAsync = promisify(execFile);

const LANG = {
  en: "Respond in English.",
  ur: "Respond in Urdu (اردو script).",
  roman: "Respond in Roman Urdu (Urdu written in Latin script).",
};

function schemaHint() {
  return `Return ONLY valid JSON (no markdown fences) with this structure:
{
  "shot_size": "e.g. wide shot | medium shot | close-up | extreme close-up | full shot",
  "camera_angle": "e.g. eye-level | low-angle | high-angle | top-down | over-the-shoulder | dutch",
  "camera_height": "e.g. ground level | knee | waist | chest | eye level | elevated | aerial",
  "camera_direction": "direction the camera faces relative to subject, if observable",
  "camera_movement": "describe start position, direction, speed, height changes, end framing, and how it connects to the next shot",
  "movement_speed": "static | slow | moderate | fast",
  "subject_placement": "e.g. centered | rule-of-thirds left | rule-of-thirds right",
  "foreground": "...", "middleground": "...", "background": "...",
  "depth_of_field": "shallow | medium | deep, and what is in focus",
  "focus_target": "...",
  "composition_notes": "symmetry, leading lines, negative space, foreground framing, etc.",
  "characters": [{"label":"CHARACTER_A","age_range":"...","hair":"...","clothing":"...","distinctive_features":"...","position":"...","posture":"...","movement":"...","screen_position":"...","expression":"..."}],
  "objects": [{"label":"OBJECT_A","description":"...","position":"..."}],
  "actions": ["ordered list of what happens, in sequence"],
  "environment": {"location_type":"...","time_of_day":"...","weather":"...","lighting_direction":"...","light_quality":"...","shadows":"...","atmosphere":"...","surfaces":"..."},
  "color_palette": "...", "color_grading": "...",
  "dialogue": [{"start":0.0,"end":0.0,"speaker":"CHARACTER_A","text":"..."}],
  "sounds": ["ambient / effects actually audible or visible"],
  "transition_to_next": "cut | fade | dissolve | continuous action | uncertain",
  "continuity_notes": "what must stay identical in a recreation (clothing, object positions, lighting...)",
  "confidence": {"camera_angle":0-1,"camera_movement":0-1,"character_tracking":0-1,"action":0-1,"lighting":0-1,"dialogue":0-1},
  "uncertain": ["list anything you could not determine reliably"]
}
Rules: never invent dialogue, characters, weather, or effects you cannot see/hear. Use "uncertain" for anything ambiguous. Reuse CHARACTER_A/CHARACTER_B labels consistently for the same person across scenes. Describe only what is visible — do not redesign the scene into animation or a different grade.`;
}

function buildPrompt(scene, cameraHint, language, hasTranscript) {
  return `You are a cinematic shot analyst. Analyze these ${scene.frames.length} frames from ONE continuous shot of a video (timestamps ${scene.start}s–${scene.end}s, duration ${scene.duration}s).

PIXEL-BASED CAMERA HINT (from motion estimation — verify against the frames, do not blindly trust): ${cameraHint.movement} — ${cameraHint.detail}. IMPORTANT: if a subject moves toward a stationary camera, that is NOT a camera push-in. Only report camera movement you can actually see.

${hasTranscript ? "A speech transcript was extracted separately; only include dialogue you can verify." : "No reliable transcript is available; set dialogue to [] unless speech is clearly lip-readable, and list it under uncertain."}

${LANG[language] || LANG.en}

${schemaHint()}`;
}

async function frameToBase64(framePath, maxW = 768) {
  const tmp = framePath + ".small.jpg";
  try {
    await execFileAsync("ffmpeg", ["-y", "-v", "error", "-i", framePath, "-vf", `scale=${maxW}:-1`, "-q:v", "4", tmp]);
    const data = await fsp.readFile(fs.existsSync(tmp) ? tmp : framePath);
    return data.toString("base64");
  } finally {
    await fsp.unlink(tmp).catch(() => {});
  }
}

function cleanJson(text) {
  const t = text.trim().replace(/^```json\s*/i, "").replace(/^```\s*/, "").replace(/\s*```$/, "");
  return JSON.parse(t);
}

async function callOpenAI(framesB64, prompt, model, apiKey) {
  const content = [{ type: "text", text: prompt }];
  for (const b64 of framesB64) content.push({ type: "image_url", image_url: { url: `data:image/jpeg;base64,${b64}` } });
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model, temperature: 0.2, response_format: { type: "json_object" },
      messages: [
        { role: "system", content: "You are a precise cinematic shot analyst. Evidence over creativity. Mark uncertainty explicitly." },
        { role: "user", content },
      ],
    }),
  });
  if (!res.ok) throw new Error(`OpenAI vision failed (${res.status}): ${(await res.text()).slice(0, 300)}`);
  const data = await res.json();
  return cleanJson(data.choices[0].message.content);
}

async function callGemini(framesB64, prompt, model, apiKey) {
  const parts = [{ text: prompt }];
  for (const b64 of framesB64) parts.push({ inline_data: { mime_type: "image/jpeg", data: b64 } });
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(apiKey)}`;
  const res = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ contents: [{ parts }], generationConfig: { responseMimeType: "application/json", temperature: 0.2 } }),
  });
  if (!res.ok) throw new Error(`Gemini vision failed (${res.status}): ${(await res.text()).slice(0, 300)}`);
  const data = await res.json();
  const text = data.candidates?.[0]?.content?.parts?.map((p) => p.text || "").join("") || "";
  return cleanJson(text);
}

async function analyzeSceneVision(scene, language, cameraHint, hasTranscript) {
  const provider = (process.env.VISION_PROVIDER || "none").toLowerCase();
  if (provider === "none") return null;
  const prompt = buildPrompt(scene, cameraHint, language, hasTranscript);
  const framesB64 = [];
  for (const f of scene.frames) framesB64.push(await frameToBase64(f.path));
  if (provider === "openai") {
    if (!process.env.OPENAI_API_KEY) throw new Error("VISION_PROVIDER=openai but OPENAI_API_KEY is not set.");
    return callOpenAI(framesB64, prompt, process.env.OPENAI_VISION_MODEL || "gpt-4o", process.env.OPENAI_API_KEY);
  }
  if (provider === "gemini") {
    if (!process.env.GEMINI_API_KEY) throw new Error("VISION_PROVIDER=gemini but GEMINI_API_KEY is not set.");
    return callGemini(framesB64, prompt, process.env.GEMINI_VISION_MODEL || "gemini-2.0-flash", process.env.GEMINI_API_KEY);
  }
  throw new Error(`Unknown VISION_PROVIDER "${provider}". Use openai, gemini, or none.`);
}

module.exports = { analyzeSceneVision };
