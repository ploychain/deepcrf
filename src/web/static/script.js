async function startGame() {
  const res = await fetch('/start', { method: 'POST' });
  const data = await res.json();
  renderState(data);
}

async function act(i) {
  const res = await fetch('/act', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action_id: i })
  });
  const data = await res.json();
  renderState(data);
}

function renderState(s) {
  document.getElementById('pot').innerText = `底池: ${s.pot.toFixed(2)}`;
  document.getElementById('board').innerText = s.board.join(' ') || "暂无公共牌";

  document.getElementById('ai-chips').innerText = `筹码: ${s.players[1].stack.toFixed(2)}`;
  document.getElementById('human-chips').innerText = `筹码: ${s.players[0].stack.toFixed(2)}`;

  const actionsDiv = document.getElementById('actions');
  actionsDiv.innerHTML = '';

  if (s.final_state) {
    document.getElementById('info').innerHTML = `<h2>🏆 赢家: Player ${s.winner.join(', ')}</h2>`;
    return;
  }

  if (s.legal_actions.length > 0) {
    s.legal_actions.forEach((a, i) => {
      const btn = document.createElement('button');
      btn.innerText = a;
      btn.onclick = () => act(i);
      actionsDiv.appendChild(btn);
    });
  }
}

window.onload = () => {
  const startBtn = document.createElement('button');
  startBtn.innerText = '开始新游戏';
  startBtn.onclick = startGame;
  document.body.insertBefore(startBtn, document.getElementById('table'));
};
