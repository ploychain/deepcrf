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

  // ---- 牌局阶段 ----
  const stage = document.getElementById("stageInfo");
  if (stage) {
    const stageLabel = s.stage ? s.stage.toString() : "";
    const prettyStage = stageLabel.replace("StateStatus.", "") || "未知阶段";
    stage.textContent = `阶段: ${prettyStage}`;
  }

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
  <div class="fold-banner">弃牌</div>
  <div class="name-row">
    <div class="name">${i === 0 ? "你 (Player 0)" : "AI 玩家 " + i}</div>
    <div class="badges"></div>
  </div>
  <div class="stack">
    ${p.active === false ? "（已弃牌）" : `筹码: ${(p.stack ?? 0).toFixed(2)}`}
  </div>
  <div class="bet"></div>
  <div class="hand"></div>
  <div class="status"></div>
  <div class="last-action"></div>`;
    const h = seat.querySelector(".hand");
    const status = seat.querySelector(".status");
    const bet = seat.querySelector(".bet");
    const badges = seat.querySelector(".badges");
    const lastAction = seat.querySelector(".last-action");
    const foldBanner = seat.querySelector(".fold-banner");
    const cards = p.hand || [];

    // ✅ 新增弃牌显示逻辑
    if (!p.active) {
      seat.classList.add("folded");
      h.innerHTML = "";            // 清空手牌
      status.textContent = "弃牌"; // 显示弃牌文字
      if (foldBanner) foldBanner.style.display = "flex";
    } else {
      seat.classList.remove("folded");
      status.textContent = "";
      if (foldBanner) foldBanner.style.display = "none";
    }

    // ✅ 新增：弃牌状态显示
    if (!p.active) seat.classList.add("folded");
    else seat.classList.remove("folded");

    // ✅ 新增：盲注/庄家徽章
    if (badges) {
      badges.innerHTML = "";
      const addBadge = (txt, cls, title) => {
        const tag = document.createElement("span");
        tag.className = `badge ${cls || ""}`.trim();
        tag.textContent = txt;
        if (title) tag.title = title;
        badges.appendChild(tag);
        return tag;
      };
      if (p.is_dealer) addBadge("D", "dealer", "庄家");
      if (p.is_small_blind) {
        const amt = typeof s.small_blind_amount === "number" ? s.small_blind_amount : null;
        addBadge("SB", "sb", amt ? `小盲 ${amt.toFixed(2)}` : "小盲");
      }
      if (p.is_big_blind) {
        const amt = typeof s.big_blind_amount === "number" ? s.big_blind_amount : null;
        addBadge("BB", "bb", amt ? `大盲 ${amt.toFixed(2)}` : "大盲");
      }
    }

    // ✅ 新增：显示本轮投注
    if (bet) {
      const amount = typeof p.bet === "number" ? p.bet : 0;
      bet.textContent = amount > 0 ? `本轮投注: ${amount.toFixed(2)}` : "";
    }

    // ✅ 新增：显示上一动作或结算
    if (lastAction) {
      if (s.final_state && typeof p.reward === "number" && p.reward !== 0) {
        const sign = p.reward > 0 ? "+" : "";
        lastAction.textContent = `结算: ${sign}${p.reward.toFixed(2)}`;
      } else if (p.last_action) {
        lastAction.textContent = `动作: ${p.last_action}`;
      } else {
        lastAction.textContent = "";
      }
    }

    // ✅ 新增：摊牌逻辑
    const showCards = cardList => {
      cardList.forEach(txt => {
        const d = document.createElement("div");
        d.className = "card";
        const val = normalizeCard(txt);
        d.textContent = val;
        if (val.includes("♥") || val.includes("♦")) d.classList.add("red");
        h.appendChild(d);
      });
    };

    if (i === 0 && cards.length > 0) {
      // 玩家自己始终看到手牌
      showCards(cards);
    } else if (s.final_state && p.active && cards.length > 0) {
      // 牌局结束，仅展示仍在桌上玩家
      showCards(cards);
    } else if (!p.active) {
      // 弃牌玩家不展示手牌
      h.innerHTML = "";
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
  if (startBtn) {
    startBtn.onclick = startGame;
    startGame();
  } else startGame();
};
