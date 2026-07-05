const API_BASE = window.location.origin;

const chatWindow = document.getElementById('chatWindow');
const promptInput = document.getElementById('promptInput');
const sendBtn = document.getElementById('sendBtn');
const strategySelect = document.getElementById('strategy');
const maxTokensInput = document.getElementById('maxTokens');
const temperatureInput = document.getElementById('temperature');
const tempValueLabel = document.getElementById('tempValue');
const tokenizerPanel = document.getElementById('tokenizerPanel');
const attentionPanel = document.getElementById('attentionPanel');
const statsBar = document.getElementById('statsBar');
const compareBtn = document.getElementById('compareBtn');
const compareModal = document.getElementById('compareModal');
const closeCompareBtn = document.getElementById('closeCompareBtn');
const runCompareBtn = document.getElementById('runCompareBtn');
const strategyASelect = document.getElementById('strategyA');
const strategyBSelect = document.getElementById('strategyB');
const labelA = document.getElementById('labelA');
const labelB = document.getElementById('labelB');
const resultA = document.getElementById('resultA');
const resultB = document.getElementById('resultB');

function temperatureLabel(value) {
  if (value <= 0.4) return `${value} — safe`;
  if (value <= 0.7) return `${value} — balanced`;
  if (value <= 1.0) return `${value} — creative`;
  return `${value} — wild`;
}

temperatureInput.addEventListener('input', () => {
  tempValueLabel.textContent = temperatureLabel(temperatureInput.value);
});

function probToColor(prob) {
  const hue = Math.round(prob * 120);
  return `hsl(${hue}, 70%, 55%)`;
}

function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

async function showTokenizerBreakdown(text) {
  if (!text.trim()) {
    tokenizerPanel.innerHTML = '';
    return;
  }
  try {
    const res = await fetch(`${API_BASE}/api/tokenize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    tokenizerPanel.innerHTML = data.pieces
      .map((p) => `<span class="token-piece" title="id: ${p.id}">${escapeHtml(p.text)}</span>`)
      .join('');
  } catch (err) {
    tokenizerPanel.innerHTML = '';
  }
}

let debounceTimer = null;
promptInput.addEventListener('input', () => {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => showTokenizerBreakdown(promptInput.value), 300);
});

function renderAttentionHeatmap(pieces, weights) {
  if (!pieces || !weights || pieces.length === 0) {
    attentionPanel.innerHTML = '';
    return;
  }
  const max = Math.max(...weights, 1e-9);
  attentionPanel.innerHTML = pieces
    .map((p, i) => {
      const intensity = weights[i] / max;
      const alpha = 0.15 + intensity * 0.65;
      return `<span class="attn-piece" style="background: rgba(224,165,66,${alpha.toFixed(2)})" title="attention: ${(weights[i] * 100).toFixed(1)}%">${escapeHtml(p.text)}</span>`;
    })
    .join('');
}

function appendMessage(role, text) {
  const div = document.createElement('div');
  div.className = `message ${role}`;
  div.textContent = text;
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return div;
}

async function sendPrompt() {
  const prompt = promptInput.value.trim();
  if (!prompt) return;

  appendMessage('user', prompt);
  promptInput.value = '';
  tokenizerPanel.innerHTML = '';
  sendBtn.disabled = true;

  const assistantDiv = document.createElement('div');
  assistantDiv.className = 'message assistant';
  const textSpan = document.createElement('span');
  const candidatesRow = document.createElement('div');
  candidatesRow.className = 'candidates-row';
  candidatesRow.style.display = 'none';
  assistantDiv.appendChild(textSpan);
  assistantDiv.appendChild(candidatesRow);
  chatWindow.appendChild(assistantDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;

  try {
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        strategy: strategySelect.value,
        max_new_tokens: Number(maxTokensInput.value),
        temperature: Number(temperatureInput.value),
      }),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = JSON.parse(line.slice(6));

        if (data.promptPieces) {
          renderAttentionHeatmap(data.promptPieces, data.promptAttention);
          continue;
        }

        if (data.token) {
          const span = document.createElement('span');
          span.className = 'token-span';
          span.textContent = data.token;
          span.style.color = probToColor(data.prob);
          span.title = `probability: ${(data.prob * 100).toFixed(1)}%`;
          textSpan.appendChild(span);

          if (data.candidates) {
            candidatesRow.style.display = 'flex';
            candidatesRow.innerHTML = data.candidates
              .map((c) => {
                const isChosen = c.text.trim() === data.token.trim();
                return `<span class="candidate-chip${isChosen ? ' chosen' : ''}">${escapeHtml(c.text)} ${(c.prob * 100).toFixed(0)}%</span>`;
              })
              .join('');
          }

          if (data.tokensPerSec !== undefined) {
            statsBar.textContent = `${data.tokensPerSec.toFixed(1)} tok/s · ${data.elapsed.toFixed(1)}s elapsed`;
          }

          chatWindow.scrollTop = chatWindow.scrollHeight;
        }
      }
    }
  } catch (err) {
    textSpan.textContent = `Error: ${err.message}`;
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener('click', sendPrompt);
promptInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendPrompt();
});

compareBtn.addEventListener('click', () => {
  compareModal.classList.remove('hidden');
});
closeCompareBtn.addEventListener('click', () => {
  compareModal.classList.add('hidden');
});

strategyASelect.addEventListener('change', () => {
  labelA.textContent = strategyASelect.options[strategyASelect.selectedIndex].text;
});
strategyBSelect.addEventListener('change', () => {
  labelB.textContent = strategyBSelect.options[strategyBSelect.selectedIndex].text;
});

runCompareBtn.addEventListener('click', async () => {
  const prompt = promptInput.value.trim() || 'The future of technology is';
  runCompareBtn.disabled = true;
  resultA.textContent = 'Generating...';
  resultB.textContent = 'Generating...';

  try {
    const res = await fetch(`${API_BASE}/api/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        strategyA: strategyASelect.value,
        strategyB: strategyBSelect.value,
        max_new_tokens: 40,
        seed: 42,
      }),
    });
    const data = await res.json();
    resultA.textContent = data.resultA;
    resultB.textContent = data.resultB;
  } catch (err) {
    resultA.textContent = `Error: ${err.message}`;
  } finally {
    runCompareBtn.disabled = false;
  }
});
