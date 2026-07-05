const terminalBody = document.getElementById('terminalBody');
const introStartBtn = document.getElementById('introStartBtn');
const introOverlay = document.getElementById('introOverlay');

const BOOT_STEPS = [
  { text: '$ loading byte-level BPE tokenizer...', delay: 400 },
  { text: '  vocab size: 50,257 tokens', delay: 250, dim: true },
  { text: '$ loading GPT-2 124M weights (safetensors)...', delay: 500 },
  { text: '  12 layers · 12 attention heads · 768 dims', delay: 250, dim: true },
  { text: '$ initializing causal self-attention...', delay: 400 },
  { text: '$ building KV cache...', delay: 350 },
  { text: '$ connecting to inference backend...', delay: 400 },
];

function appendLine(text, opts = {}) {
  const div = document.createElement('div');
  div.className = 'terminal-line' + (opts.ok ? ' terminal-ok' : '') + (opts.dim ? ' terminal-dim' : '');
  div.textContent = text;
  terminalBody.appendChild(div);
  return div;
}

async function checkBackendReady() {
  try {
    const res = await fetch(`${window.location.origin}/api/health`);
    return res.ok;
  } catch {
    return false;
  }
}

async function runBootSequence() {
  for (const step of BOOT_STEPS) {
    await new Promise((r) => setTimeout(r, step.delay));
    appendLine(step.text, { dim: step.dim });
  }

  const statusLine = appendLine('$ checking model status...');
  const cursor = document.createElement('span');
  cursor.className = 'terminal-cursor';
  statusLine.appendChild(cursor);

  let attempts = 0;
  let ready = false;
  while (attempts < 30 && !ready) {
    ready = await checkBackendReady();
    if (!ready) {
      await new Promise((r) => setTimeout(r, 1000));
      attempts++;
    }
  }

  cursor.remove();

  if (ready) {
    appendLine('✓ model ready — all systems operational', { ok: true });
    introStartBtn.classList.remove('hidden');
  } else {
    appendLine('✗ backend not responding — check server logs', {});
  }
}

introStartBtn.addEventListener('click', () => {
  introOverlay.classList.add('hidden');
  document.getElementById('promptInput').focus();
});

runBootSequence();
