const bgCanvas = document.getElementById('bgCanvas');
const bgCtx = bgCanvas.getContext('2d');

function resizeBg() {
  bgCanvas.width = window.innerWidth;
  bgCanvas.height = window.innerHeight;
}
window.addEventListener('resize', resizeBg);
resizeBg();

const NODE_COUNT = 70;
const nodes = Array.from({ length: NODE_COUNT }, () => ({
  x: Math.random() * bgCanvas.width,
  y: Math.random() * bgCanvas.height,
  vx: (Math.random() - 0.5) * 0.3,
  vy: (Math.random() - 0.5) * 0.3,
  hue: Math.random() > 0.5 ? 175 : 300,
}));

function step() {
  bgCtx.fillStyle = 'rgba(5, 7, 13, 1)';
  bgCtx.fillRect(0, 0, bgCanvas.width, bgCanvas.height);

  for (const n of nodes) {
    n.x += n.vx;
    n.y += n.vy;
    if (n.x < 0 || n.x > bgCanvas.width) n.vx *= -1;
    if (n.y < 0 || n.y > bgCanvas.height) n.vy *= -1;
  }

  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], b = nodes[j];
      const dist = Math.hypot(a.x - b.x, a.y - b.y);
      if (dist < 150) {
        bgCtx.strokeStyle = `hsla(${(a.hue + b.hue) / 2}, 80%, 65%, ${0.1 * (1 - dist / 150)})`;
        bgCtx.lineWidth = 1;
        bgCtx.beginPath();
        bgCtx.moveTo(a.x, a.y);
        bgCtx.lineTo(b.x, b.y);
        bgCtx.stroke();
      }
    }
  }

  for (const n of nodes) {
    bgCtx.fillStyle = `hsla(${n.hue}, 90%, 65%, 0.5)`;
    bgCtx.beginPath();
    bgCtx.arc(n.x, n.y, 1.5, 0, Math.PI * 2);
    bgCtx.fill();
  }

  requestAnimationFrame(step);
}
step();
