let currentState = null;

async function startGame() {
  const res = await fetch("/start", { method: "POST" });
  const data = await res.json();
  currentState = data;
  renderState(data);
}

async function act(actionIndex) {
  if (!currentState) return;

  const res = await fetch("/act", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action_id: actionIndex })
  });

  const data = await res.json();
  currentState = data;
  renderState(data);
}

/* ======== 渲染游戏状态 ======== */
function renderState(state) {
  const table = document.getElementById("table");
  table.innerHTML = ""; // 清空桌面

  // --- 公共牌 ---
  const communityDiv = document.createElement("div");
  communityDiv.id = "community";
  state.board.forEach(c => {
    const card = document.createElement("div");
    card.className = "card";
    card.textContent = c;
    communityDiv.appendChild(card);
  });
  table.appendChild(communityDiv);

  // --- 底池 ---
  const potDiv = document.createElement("div");
  potDiv.id = "pot";
  potDiv.textContent = "底池: " + state.pot.toFixed(2);
  table.appendChild(potDiv);

  // --- 玩家 ---
  state.players.forEach(p => {
    const playerDiv = document.createElement("div");
    playerDiv.className = "player";
    playerDiv.id = "player-" + p.id;
    if (p.active) playerDiv.classList.add("active");

    const name = document.createElement("div");
    name.className = "name";
    name.textContent = p.id === 0 ? "你 (Player 0)" : "AI 玩家 " + p.id;

    const stack = document.createElement("div");
    stack.className = "stack";
    stack.textContent = "筹码: " + p.stack.toFixed(2);

    const handDiv = document.createElement("div");
    handDiv.className = "hand";
    p.hand.forEach(cardStr => {
      const card = document.createElement("div");
      card.className = "card";
      if (cardStr === "🂠") card.classList.add("card-back");
      card.textContent = cardStr;
      handDiv.appendChild(card);
    });

    playerDiv.appendChild(name);
    playerDiv.appendChild(stack);
    playerDiv.appendChild(handDiv);

    table.appendChild(playerDiv);
  });

  // --- 动作按钮（仅你能操作） ---
  if (!state.final_state && state.current_player === 0) {
    const controls = document.createElement("div");
    controls.id = "controls";
    state.legal_actions.forEach((action, idx) => {
      const btn = document.createElement("button");
      btn.textContent = action.replace("ActionEnum.", "");
      btn.onclick = () => act(idx);
      controls.appendChild(btn);
    });
    table.appendChild(controls);
  }

  // --- 若已结束 ---
  if (state.final_state) {
    const msg = document.createElement("div");
    msg.style.position = "absolute";
    msg.style.top = "45%";
    msg.style.left = "50%";
    msg.style.transform = "translate(-50%, -50%)";
    msg.style.fontSize = "36px";
    msg.style.fontWeight = "bold";
    msg.style.color = "#ffeb3b";
    msg.textContent = state.winner.includes(0) ? "🎉 你赢了！" : "😢 AI 赢了！";
    table.appendChild(msg);
  }
}

window.onload = startGame;
