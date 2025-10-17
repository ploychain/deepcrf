# src/web/server.py
from flask import Flask, jsonify, request, send_from_directory
import torch
import os
from pokers import State, StateStatus, ActionEnum
from src.core.deep_cfr import DeepCFRAgent
from src.agents.random_agent import RandomAgent

app = Flask(__name__, static_folder="static", static_url_path="/static")

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/checkpoint_iter_100.pt"

# ---------- 初始化AI ----------
def safe_load_agent():
    """加载AI模型，如果失败则启用随机AI"""
    try:
        print(f"🔹 正在加载AI模型：{MODEL_PATH}")
        agent = DeepCFRAgent(player_id=0, num_players=6, device=device)
        ckpt = torch.load(MODEL_PATH, map_location=device)
        try:
            agent.advantage_net.load_state_dict(ckpt["advantage_net"], strict=False)
            agent.strategy_net.load_state_dict(ckpt["strategy_net"], strict=False)
            print("✅ 模型加载成功")
        except Exception as e:
            print(f"⚠️ 模型结构不匹配: {e}")
            print("⚠️ 启用随机策略 AI 替代")
            agent = RandomAgent(0)
        return agent
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("⚠️ 启用随机策略 AI 替代")
        return RandomAgent(0)

AI_AGENT = safe_load_agent()
CURRENT_STATE = None


# ---------- 游戏状态转JSON ----------
def serialize_state(state):
    """把游戏状态打包成前端需要的 JSON（鲁棒地把 Card → 'A♠' 这种）"""

    # ----- 工具：rank/suit 映射 -----
    RANK_STR = {1:"A", 14:"A", 13:"K", 12:"Q", 11:"J", 10:"T",
                9:"9", 8:"8", 7:"7", 6:"6", 5:"5", 4:"4", 3:"3", 2:"2"}
    SUIT_STR = {
        "S":"♠", "SPADE":"♠", "SPADES":"♠", 3:"♠",
        "H":"♥", "HEART":"♥", "HEARTS":"♥", 2:"♥",
        "D":"♦", "DIAMOND":"♦", "DIAMONDS":"♦", 1:"♦",
        "C":"♣", "CLUB":"♣", "CLUBS":"♣", 0:"♣",
    }
    SUITS_52 = {0:"♣", 1:"♦", 2:"♥", 3:"♠"}  # 用于0..51编码
    RANKS_52 = {0:"2",1:"3",2:"4",3:"5",4:"6",5:"7",6:"8",7:"9",8:"T",9:"J",10:"Q",11:"K",12:"A"}

    def card_to_unicode(c):
        """最大兼容把一张牌转成 'A♠' 这类文本"""
        # 字符串本身（已经是 "A♠" 或 "🂠"）
        if isinstance(c, str):
            return c

        # 0..51 的整型索引
        if isinstance(c, int) and 0 <= c <= 51:
            suit = c // 13
            rank = c % 13
            return f"{RANKS_52.get(rank,'?')}{SUITS_52.get(suit,'?')}"

        # dict 形式 {"rank":...,"suit":...}
        if isinstance(c, dict):
            r = c.get("rank"); s = c.get("suit")
            return f"{_rank_to_str(r)}{_suit_to_str(s)}"

        # 一般对象：尽量读 rank / suit / label
        r = getattr(c, "rank", None)
        s = getattr(c, "suit", None)
        if r is not None and s is not None:
            return f"{_rank_to_str(r)}{_suit_to_str(s)}"

        lab = getattr(c, "label", None)
        if lab:
            return str(lab)

        # 兜底：str(c)（如果还是 <builtins.Card …>，前端会显示 ?）
        text = str(c)
        if text.startswith("<") and "object at" in text:
            return "?"
        return text

    def _rank_to_str(r):
        # Enum / 对象：优先 name / value
        name = getattr(r, "name", None)
        if name:
            return RANK_STR.get(_maybe_int(name), name)
        val = getattr(r, "value", None)
        if val is not None:
            return RANK_STR.get(_maybe_int(val), str(val))
        # 直接 int/str
        if isinstance(r, int):
            return RANK_STR.get(r, str(r))
        if isinstance(r, str):
            # "A","K","Q","J","T","2".. 直接返回
            return r.upper()
        return "?"

    def _suit_to_str(s):
        name = getattr(s, "name", None)
        if name:
            return SUIT_STR.get(name.upper(), name)
        val = getattr(s, "value", None)
        if val is not None:
            return SUIT_STR.get(val, str(val))
        if isinstance(s, str):
            return SUIT_STR.get(s.upper(), s)
        if isinstance(s, int):
            return SUIT_STR.get(s, str(s))
        return "?"

    def _maybe_int(x):
        try:
            return int(x)
        except Exception:
            return x

    # --- 取公共区（兼容 board/community） ---
    community = getattr(state, "community", None)
    if community is None:
        community = getattr(state, "board", [])

    # --- legal actions 文本化 ---
    legal_acts = []
    for a in getattr(state, "legal_actions", []):
        nm = getattr(a, "name", None)
        legal_acts.append(nm if nm else str(a))

    # --- players / 手牌（只展示玩家0的） ---
    players = []
    for i, ps in enumerate(state.players_state):
        raw_hand = getattr(ps, "hand", [])
        hand = [card_to_unicode(c) for c in raw_hand] if i == 0 else ["🂠", "🂠"]
        players.append({
            "id": i,
            "stack": getattr(ps, "stack", 0.0),
            "bet": getattr(ps, "bet", 0.0),
            "active": getattr(ps, "active", True),
            "hand": hand
        })

    data = {
        "board": [card_to_unicode(c) for c in community],
        "pot": float(getattr(state, "pot", 0.0)),
        "current_player": int(getattr(state, "current_player", 0)),
        "final_state": bool(getattr(state, "final_state", False)),
        "legal_actions": legal_acts if (not getattr(state, "final_state", False) and int(getattr(state, "current_player", 0)) == 0) else [],
        "players": players,
        "winner": []
    }

    if data["final_state"]:
        data["winner"] = [
            i for i, ps in enumerate(state.players_state)
            if float(getattr(ps, "reward", 0.0)) > 0.0
        ]

    return data



# ---------- 路由 ----------
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/start", methods=["POST"])
def start():
    global CURRENT_STATE
    print("♠️ 开始新局")

    import random
    # 初始化一局游戏
    CURRENT_STATE = State.from_seed(
        n_players=6,
        button=random.randint(0, 5),  # 随机庄家位置
        sb=1,
        bb=2,
        stake=200.0,
        seed=random.randint(0, 10000)  # 随机种子确保每局不同
    )

    return jsonify(serialize_state(CURRENT_STATE))


@app.route("/act", methods=["POST"])
def act():
    global CURRENT_STATE

    if not CURRENT_STATE:
        return jsonify({"error": "Game not started"}), 400

    # 获取玩家动作
    data = request.get_json()
    action_id = data.get("action_id", 0)
    legal = CURRENT_STATE.legal_actions
    if not legal:
        return jsonify({"error": "No legal actions"}), 400

    # 玩家执行动作
    try:
        player_action = legal[action_id]
    except IndexError:
        player_action = legal[0]

    print(f"你执行动作: {player_action}")
    CURRENT_STATE = CURRENT_STATE.apply_action(player_action)

    # 如果AI能动，AI自动连续行动
    while (not CURRENT_STATE.final_state and
           CURRENT_STATE.current_player != 0 and
           len(CURRENT_STATE.legal_actions) > 0):

        ai_action = AI_AGENT.choose_action(CURRENT_STATE)
        print(f"AI 玩家 {CURRENT_STATE.current_player} 执行动作: {ai_action}")
        new_state = CURRENT_STATE.apply_action(ai_action)

        # 防止错误状态卡死
        if new_state.status != StateStatus.Ok:
            print(f"⚠️ 非法动作：{new_state.status}")
            break

        CURRENT_STATE = new_state

    return jsonify(serialize_state(CURRENT_STATE))


if __name__ == "__main__":
    port = 5000
    print(f"✅ Poker Web 服务器启动中：http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
