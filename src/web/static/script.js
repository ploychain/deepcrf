// ===================== 启动游戏 =====================
async function startGame() {
  try {
    const res = await fetch("/start", { method: "POST" });
    const s = await res.json();
    console.log("✅ 游戏开始:", s);
    renderState(s);
  } catch (err) {
    console.error("❌ startGame error:", err);
  }
}

// ===================== 玩家动作 =====================
async function act(action_id) {
  try {
    const res = await fetch("/act", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action_id })
    });
    const s = await res.json();
    console.log("✅ 玩家动作:", s);
    renderState(s);
  } catch (err) {
    console.error("❌ act error:", err);
  }
}

// ===================== 渲染游戏状态 =====================
function renderState(s) {
  if (!s) {
    console.error("❌ renderState received null state");
    return;
  }

  // ---- 公共牌 ----
  const comm = document.getElementById("community");
  comm.innerHTML = "";
  const board = s.community || [];
  board.forEach(txt => {
    const c = document.createElement("div");
    c.className = "card";
    c.textContent = normalizeCard(txt);
    comm.appendChild(c);
  });

  // ---- 底池 ----
  const pot = document.getElementById("pot");
  pot.textContent = "底池: " + (s.pot ? s.pot.toFixed(2) : "0");

  // ---- 玩家 ----
  if (!s.players || !Array.isArray(s.players)) {
    console.error("❌ s.players invalid:", s);
    return;
  }

  s.players.forEach((p, i) => {
    const seat = document.getElementById("player-" + i);
    if (!seat) return;

    // ✅ 保留原始结构
    seat.innerHTML = `
  <div class="name">${i === 0 ? "你 (Player 0)" : "AI 玩家 " + i}</div>
  <div class="stack">
  ${p.active === false ? "（已弃牌）" : `筹码: ${(p.stack ?? 0).toFixed(2)}`}
</div>
  <div class="hand"></div>
  <div class="status"></div>`;
    const h = seat.querySelector(".hand");
    const status = seat.querySelector(".status");
    const cards = p.hand || [];

    // ✅ 新增弃牌显示逻辑
    if (!p.active) {
      seat.classList.add("folded");
      h.innerHTML = "";            // 清空手牌
      status.textContent = "弃牌"; // 显示弃牌文字
    } else {
      seat.classList.remove("folded");
      status.textContent = "";
    }

    // ✅ 新增：弃牌状态显示
    if (!p.active) seat.classList.add("folded");
    else seat.classList.remove("folded");

    // ✅ 新增：摊牌逻辑
    if (s.final_state && cards.length > 0 && p.active) {
      // 游戏结束时翻开所有仍在局内的手牌
      cards.forEach(txt => {
        const d = document.createElement("div");
        d.className = "card";
        const val = normalizeCard(txt);
        d.textContent = val;
        if (val.includes("♥") || val.includes("♦")) d.classList.add("red");
        h.appendChild(d);
      });
    } else if (i === 0 || s.final_state) {
      // ✅ 玩家自己 或 牌局结束时 — 显示真实手牌
      cards.forEach(txt => {
        const d = document.createElement("div");
        d.className = "card";
        const val = normalizeCard(txt);
        d.textContent = val;
        if (val.includes("♥") || val.includes("♦")) d.classList.add("red");
        h.appendChild(d);
      });
    } else {
      // ✅ 进行中时的 AI 玩家 — 显示背面
      for (let j = 0; j < 2; j++) {
        const d = document.createElement("div");
        d.className = "card card-back";
        d.textContent = "🂠";
        h.appendChild(d);
      }
    }

    // 当前行动玩家高亮
    if (i === s.current_player) seat.classList.add("active");
    else seat.classList.remove("active");
  });

  // ---- 控制按钮 ----
  const ctrls = document.getElementById("controls");
  ctrls.innerHTML = "";
  if (!s.final_state && s.current_player === 0 && s.legal_actions) {
    s.legal_actions.forEach((name, idx) => {
      const b = document.createElement("button");
      b.textContent = name.replace("ActionEnum.", "");
      b.onclick = () => act(idx);
      ctrls.appendChild(b);
    });
  }

  // ---- 结束提示 ----
  const end = document.getElementById("endMessage");
  if (!end) return;

  if (s.final_state) {
    end.style.display = "block";
    const isWin = s.winner && s.winner.includes(0);
    end.textContent = isWin ? "🎉 你赢了!" : "😢 AI 赢了!";
  } else {
    end.style.display = "none";
  }
}


// ===================== 扑克牌文本修复 =====================
function normalizeCard(txt) {
  if (!txt) return "??";
  // 去掉 CardRank.R 前缀与点号
  txt = txt.replaceAll("CardRank.R", "")
           .replaceAll("CardRank.", "")
           .replaceAll("R", "")
           .replaceAll(".", "")
           .trim();

  // 提取 rank 与花色
  const match = txt.match(/([0-9TJQKA]+)([♠♥♦♣])?/);
  if (match) {
    const rankMap = { T: "10", J: "J", Q: "Q", K: "K", A: "A" };
    const rank = rankMap[match[1]] || match[1];
    const suit = match[2] || "";
    return rank + suit;
  }

  return txt;
}

// ===================== 初始化 =====================
window.onload = () => {
  const startBtn = document.getElementById("startBtn");
  if (startBtn) startBtn.onclick = startGame;
  else startGame();
};
