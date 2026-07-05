const API_BASE = window.location.origin;

const chatWindow = document.getElementById('chatWindow');
const promptInput = document.getElementById('promptInput');
const sendBtn = document.getElementById('sendBtn');
const strategySelect = document.getElementById('strategy');
const maxTokensInput = document.getElementById('maxTokens');
const temperatureInput = document.getElementById('temperature');
const tempValueLabel = document.getElementById('tempValue');

temperatureInput.addEventListener('input', () => {
  tempValueLabel.textContent = temperatureInput.value;
});

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
  sendBtn.disabled = true;

  const assistantDiv = appendMessage('assistant', '');

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
        if (data.token) {
          assistantDiv.textContent += data.token;
          chatWindow.scrollTop = chatWindow.scrollHeight;
        }
      }
    }
  } catch (err) {
    assistantDiv.textContent = `Error: ${err.message}`;
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener('click', sendPrompt);
promptInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendPrompt();
});
