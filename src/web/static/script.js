async function startGame() {
  const res = await fetch("/start", { method: "POST" });
  const s = await res.json();
  renderState(s);
}

async function act(action_id) {
  const res = await fetch("/act", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action_id })
  });
  const s = await res.json();
  renderState(s);
}

function renderState(s) {
  const comm = document.getElementById("community");
  comm.innerHTML = "";
  s.board.forEach(txt => {
    const c = document.createElement("div");
    c.className = "card";
    c.textContent = txt;
    comm.appendChild(c);
  });

  document.getElementById("pot").textContent = "底池: " + s.pot.toFixed(2);

  for (let i = 0; i < s.players.length; i++) {
    const p = s.players[i];
    const seat = document.getElementById("player-" + i);
    seat.innerHTML = `
      <div class="name">${i === 0 ? "你 (Player 0)" : "AI 玩家 " + i}</div>
      <div class="stack">筹码: ${p.stack.toFixed(2)}</div>
      <div class="hand"></div>
    `;
    const h = seat.querySelector(".hand");
    p.hand.forEach(txt => {
      const d = document.createElement("div");
      d.className = "card" + (txt === "🂠" ? " card-back" : "");
      d.textContent = txt;
      h.appendChild(d);
    });
  }

  const ctrls = document.getElementById("controls");
  ctrls.innerHTML = "";
  if (!s.final_state && s.current_player === 0) {
    s.legal_actions.forEach((name, idx) => {
      const b = document.createElement("button");
      b.textContent = name.replace("ActionEnum.", "");
      b.onclick = () => act(idx);
      ctrls.appendChild(b);
    });
  }

  const end = document.getElementById("endMessage");
  if (s.final_state) {
    end.style.display = "block";
    end.textContent = s.winner.includes(0) ? "🎉 你赢了!" : "😢 AI 赢了!";
  } else {
    end.style.display = "none";
  }
}

// 自动开局
window.onload = startGame;
