// The desk-side loop. See the tab, rank a shelf script, keep a worse guess.
// This is a framework for the five scripts PageArm already ships. It does not
// click the page, and a rank is not proof the script is the right one.

import { promote, scoreGlance } from "./router.mjs";

export const SCRIPTS = {
  fill: "fill-sample.js",
  table: "copy-table.js",
  form: "dump-form.js",
  required: "mark-required.js",
  glance: "glance.js",
};

export function plan(glance, held) {
  const next = scoreGlance(glance);
  const picked = promote(held, next);
  return {
    family: picked.family,
    script: SCRIPTS[picked.family] || "",
    confidence: picked.confidence,
    scores: next.scores,
    promoted: !!picked.promoted,
    reason: picked.reason,
    note: "A family rank is not proof. Read the script, then save it yourself.",
  };
}
