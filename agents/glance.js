// The tab the human already has open. Names and labels, never values.
// A coding agent reads it at GET /api/glance.
// Only the top window reports. Frames would otherwise overwrite it with nothing.
agent.arm = function () {
  var topOk = false;
  try { topOk = window.top === window; } catch (e) { topOk = false; }
  if (!topOk) return;
  agent.glance();
  agent.pip("ok");
};
