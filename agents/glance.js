// The tab the human already has open. Names and labels, never values.
// A coding agent reads it at GET /api/glance.
agent.arm = function () {
  agent.glance();
  agent.pip("ok");
};
