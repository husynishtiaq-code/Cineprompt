/**
 * Stage 6: assemble measured analysis + vision output into production-ready
 * per-scene AI video prompts, formatted for each target platform.
 * Uncertain details are explicitly flagged — never presented as fact.
 */
function fmtT(s) {
  s = Math.max(0, s);
  const m = Math.floor(s / 60), sec = (s % 60);
  return `${String(m).padStart(2, "0")}:${sec.toFixed(0).padStart(2, "0")}`;
}
function fmtRange(a, b) {
  const f = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${(s % 60).toFixed(1).padStart(4, "0")}`;
  return `${f(a)}–${f(b)}`;
}
const unc = (v) => (v == null || v === "" || v === "uncertain" ? "[uncertain]" : v);

const NEGATIVE = "no extra characters, no changed clothing, no face morphing, no identity change, no beautification, no new dialogue, no invented weather or lighting effects, no slow motion unless observed, no camera movement beyond what is described, no text overlays, no watermarks";

function continuityLock(scene, vision) {
  const bits = [];
  if (vision?.characters?.length) {
    for (const c of vision.characters) {
      bits.push(`${c.label}: ${[c.hair, c.clothing, c.distinctive_features].filter(Boolean).filter((x) => x !== "uncertain").join("; ") || "appearance as in reference"}`);
    }
  }
  if (vision?.objects?.length) for (const o of vision.objects) bits.push(`${o.label} stays at: ${unc(o.position)}`);
  if (vision?.environment?.lighting_direction) bits.push(`lighting direction: ${vision.environment.lighting_direction}`);
  if (vision?.color_grading && vision.color_grading !== "uncertain") bits.push(`grade: ${vision.color_grading}`);
  if (vision?.continuity_notes) bits.push(vision.continuity_notes);
  bits.push(`duration exactly ${scene.duration}s; shot order preserved`);
  return bits;
}

function audioBlock(scene, vision, audio) {
  const lines = [];
  const segs = (audio?.transcript || []).filter((t) => t.end > scene.start && t.start < scene.end);
  if (segs.length) {
    lines.push("Verified dialogue (from transcript):");
    for (const s of segs) lines.push(`  [${fmtRange(s.start, s.end)}] "${s.text}"`);
  } else if (vision?.dialogue?.length) {
    lines.push("Dialogue (vision-observed, verify before use):");
    for (const d of vision.dialogue) lines.push(`  "${d.text}" — ${d.speaker || "unknown speaker"}`);
  } else {
    lines.push(audio?.hasAudio ? "Audio present but no verified dialogue in this shot." : "No audio / visuals only.");
  }
  if (vision?.sounds?.length) lines.push("Sound design: " + vision.sounds.join("; "));
  return lines.join("\n");
}

function confidenceBlock(scene, vision) {
  const c = vision?.confidence || {};
  const camMove = c.camera_movement ?? scene.camera?.confidence ?? null;
  const parts = [
    `camera angle ${pct(c.camera_angle)}`, `camera movement ${pct(camMove)}`,
    `shot boundary ${pct(scene.boundaryConfidence)}`, `character tracking ${pct(c.character_tracking)}`,
    `action ${pct(c.action)}`, `lighting ${pct(c.lighting)}`, `dialogue ${pct(c.dialogue)}`,
  ];
  let s = parts.join(" · ");
  const u = [...(vision?.uncertain || [])];
  if (scene.camera?.movement === "uncertain") u.push("camera movement (" + scene.camera.detail + ")");
  if (u.length) s += "\nUncertain: " + u.join("; ");
  return s;
}
const pct = (v) => (v == null ? "n/a" : Math.round(v * 100) + "%");

function referenceAnalysis(scene, vision) {
  if (!vision) return "Vision analysis unavailable for this shot (no vision provider configured). Timing, boundaries and pixel-estimated camera motion below are measured; visual details are marked [uncertain].";
  const chars = (vision.characters || []).map((c) => `${c.label} (${[c.age_range, c.hair, c.clothing].filter((x) => x && x !== "uncertain").join(", ") || "appearance as observed"})`).join("; ");
  return [
    `Shot ${scene.index} (${fmtRange(scene.start, scene.end)}, ${scene.duration}s).`,
    vision.actions?.length ? "Action sequence: " + vision.actions.join(" → ") : null,
    chars ? "Characters: " + chars : null,
    vision.environment?.location_type ? `Setting: ${vision.environment.location_type}, ${vision.environment.time_of_day || ""}`.trim() : null,
    vision.transition_to_next ? `Leads into next shot via: ${vision.transition_to_next}.` : null,
  ].filter(Boolean).join(" ");
}

function cameraBlock(scene, vision) {
  const v = vision || {};
  const est = scene.camera || {};
  const move = v.camera_movement && v.camera_movement !== "uncertain" ? v.camera_movement : `${est.movement}${est.detail ? " — " + est.detail : ""}`;
  return [
    `Shot size: ${unc(v.shot_size)}`,
    `Angle: ${unc(v.camera_angle)}; height: ${unc(v.camera_height)}; direction: ${unc(v.camera_direction)}`,
    `Movement: ${move} (speed: ${unc(v.movement_speed)})`,
    `Subject placement: ${unc(v.subject_placement)}`,
    `Depth: foreground ${unc(v.foreground)} / mid ${unc(v.middleground)} / background ${unc(v.background)}`,
    `Depth of field: ${unc(v.depth_of_field)}; focus: ${unc(v.focus_target)}`,
    v.composition_notes ? `Composition: ${v.composition_notes}` : null,
  ].filter(Boolean).join("\n");
}

function visualBlock(vision) {
  if (!vision) return "[uncertain] — vision analysis unavailable.";
  const e = vision.environment || {};
  return [
    vision.characters?.length ? "Characters: " + vision.characters.map((c) => `${c.label}: ${[c.hair, c.clothing, c.distinctive_features, c.posture, c.movement, c.expression].filter((x) => x && x !== "uncertain").join("; ")}`).join(" | ") : null,
    vision.objects?.length ? "Objects: " + vision.objects.map((o) => `${o.label} (${o.description}) at ${o.position}`).join("; ") : null,
    `Environment: ${[e.location_type, e.time_of_day, e.weather].filter((x) => x && x !== "uncertain").join(", ") || "[uncertain]"}`,
    `Lighting: ${[e.lighting_direction, e.light_quality, e.shadows].filter((x) => x && x !== "uncertain").join("; ") || "[uncertain]"}`,
    e.atmosphere && e.atmosphere !== "uncertain" ? `Atmosphere: ${e.atmosphere}` : null,
    `Palette: ${unc(vision.color_palette)}; grade: ${unc(vision.color_grading)}`,
  ].filter(Boolean).join("\n");
}

function generationPrompt(scene, vision, style) {
  // Natural-language recreation prompt
  const v = vision || {};
  const chars = (v.characters || []).map((c) =>
    `${c.label.toLowerCase().replace("_", " ")} (${[c.age_range, c.hair, c.clothing].filter((x) => x && x !== "uncertain").join(", ")})`
  ).join(" and ") || "the subject";
  const actions = v.actions?.length ? v.actions.join(", then ") : "[uncertain — motion not described]";
  const e = v.environment || {};
  const setting = [e.location_type, e.time_of_day, e.weather].filter((x) => x && x !== "uncertain").join(", ") || "the same setting as the reference";
  const light = [e.lighting_direction, e.light_quality].filter((x) => x && x !== "uncertain").join(", ") || "matching light";
  const camMove = v.camera_movement && v.camera_movement !== "uncertain" ? v.camera_movement : (scene.camera?.movement === "static" ? "Static locked-off camera" : "Camera movement as in the reference shot");
  const shot = v.shot_size && v.shot_size !== "uncertain" ? v.shot_size : "same framing as reference";
  const angle = v.camera_angle && v.camera_angle !== "uncertain" ? ` from a ${v.camera_angle}` : "";
  const base = `Recreate this exact shot (${scene.duration}s): ${shot}${angle} of ${chars} in ${setting}. ${cap(actions)}. ${camMove}. Lighting: ${light}. Photorealistic, natural physics, no new characters, no invented effects.`;
  if (style === "concise") return base;
  return base + ` Continuity: ${continuityLock(scene, v).join("; ")}.`;
}
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

/** Build the full per-scene prompt object for a target format. */
function buildScenePrompt(scene, vision, audio, target) {
  const id = `SCENE ${String(scene.index).padStart(2, "0")} | ${fmtRange(scene.start, scene.end)}`;
  const universal = [
    id,
    `REFERENCE SHOT ANALYSIS: ${referenceAnalysis(scene, vision)}`,
    `CAMERA:\n${cameraBlock(scene, vision)}`,
    `VISUAL DETAILS:\n${visualBlock(vision)}`,
    `AI VIDEO GENERATION PROMPT: ${generationPrompt(scene, vision, "full")}`,
    `CONTINUITY LOCK: ${continuityLock(scene, vision).join("; ")}`,
    `AUDIO:\n${audioBlock(scene, vision, audio)}`,
    `CONFIDENCE: ${confidenceBlock(scene, vision)}`,
  ].join("\n\n");

  let prompt, notes = "";
  switch (target) {
    case "veo":
    case "flow":
      prompt = generationPrompt(scene, vision, "full") +
        (vision?.dialogue?.length ? ` Dialogue: ${vision.dialogue.map((d) => `"${d.text}"`).join(" ")}` : "") +
        (vision?.sounds?.length ? ` Sounds: ${vision.sounds.join(", ")}.` : "") +
        ` Negative: ${NEGATIVE}.`;
      notes = target === "flow"
        ? "Paste into Google Flow (Veo). For character consistency across scenes, generate with the same ingredient/character reference."
        : "Optimized for Veo 3: descriptive paragraph, dialogue in quotes, sounds listed.";
      break;
    case "kling":
      prompt = generationPrompt(scene, vision, "concise") + `\nNegative prompt: ${NEGATIVE}.`;
      notes = "Kling prefers concise prompts — keep under ~300 words; put constraints in the negative prompt field.";
      break;
    case "seedance":
    case "hailuo":
      prompt = generationPrompt(scene, vision, "concise");
      notes = "Concise descriptive prompt; add the negative constraints in the platform's negative prompt box if available.";
      break;
    default:
      prompt = universal; notes = "Universal structured format — adapt the AI VIDEO GENERATION PROMPT block to your generator.";
  }
  return {
    scene: scene.index, start: scene.start, end: scene.end, duration: scene.duration,
    target, prompt, notes,
    sections: {
      referenceAnalysis: referenceAnalysis(scene, vision),
      camera: cameraBlock(scene, vision),
      visualDetails: visualBlock(vision),
      generationPrompt: generationPrompt(scene, vision, "full"),
      continuityLock: continuityLock(scene, vision),
      audio: audioBlock(scene, vision, audio),
      confidence: confidenceBlock(scene, vision),
    },
    frameFiles: (scene.frames || []).map((f) => f.file),
    confidenceScores: {
      camera_angle: vision?.confidence?.camera_angle ?? null,
      camera_movement: vision?.confidence?.camera_movement ?? scene.camera?.confidence ?? null,
      shot_boundary: scene.boundaryConfidence ?? null,
      character_tracking: vision?.confidence?.character_tracking ?? null,
      action: vision?.confidence?.action ?? null,
      lighting: vision?.confidence?.lighting ?? null,
      dialogue: vision?.confidence?.dialogue ?? (audio?.transcribed ? 0.8 : null),
    },
  };
}

function buildCharacterRegistry(scenesVision) {
  const reg = {};
  for (const { scene, vision } of scenesVision) {
    for (const c of vision?.characters || []) {
      if (!reg[c.label]) reg[c.label] = { label: c.label, scenes: [], hair: c.hair, clothing: c.clothing, features: c.distinctive_features };
      reg[c.label].scenes.push(scene.index);
    }
  }
  return Object.values(reg);
}

module.exports = { buildScenePrompt, buildCharacterRegistry, NEGATIVE, fmtRange };
