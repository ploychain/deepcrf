let players = [];

async function startGame() {
  const res = await fetch('/start', { method: 'POST' });
  const data = await res.json();
  renderTable(data);
}

async function act(i) {
  const res = await fetch('/act', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action_id: i })
  });
  const data = await res.json();
  renderTable(data);
}

function renderTable(s) {
  document.getElementById('board').innerText = s.board.join(' ') || "🂠🂠🂠";
  document.getElementById('pot').innerText = `底池: ${s.pot.toFixed(2)}`;

  for (let i = 0; i < s.players.length; i++) {
    const p = s.players[i];
    const seat = document.getElementById(`player-${i}`);
    if (!seat) continue;

    seat.innerHTML = `
      <div class="name">玩家 ${i}${i === 0 ? " (你)" : ""}</div>
      <div class="cards">${renderCards(i, s)}</div>
      <div class="chips">筹码: ${p.stack.toFixed(0)}</div>
    `;
  }

  const bar = document.getElementById('action-bar');
  bar.innerHTML = '';

  if (s.final_state) {
    document.getElementById('info').innerHTML =
      `<h2>🏆 赢家: 玩家 ${s.winner.join(', ')}</h2>`;
    return;
  }

  if (s.current_player === 0 && s.legal_actions.length > 0) {
    s.legal_actions.forEach((a, i) => {
      const btn = document.createElement('button');
      btn.innerText = a;
      btn.onclick = () => act(i);
      bar.appendChild(btn);
    });
  } else {
    document.getElementById('info').innerHTML = `等待 AI 动作...`;
  }
}

function renderCards(i, s) {
  if (i === 0) {
    // 玩家自己显示真实牌
    return "🂡🂢"; // TODO: 后端返回真实手牌可替换
  } else {
    return "🂠🂠";
  }
}

// 启动游戏
window.onload = () => {
  const btn = document.createElement('button');
  btn.innerText = '🎮 开始新游戏';
  btn.onclick = startGame;
  document.getElementById('action-bar').appendChild(btn);
};
