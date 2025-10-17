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
def serialize_state(state: State):
    """将游戏状态转换成可前端显示的JSON"""
    data = {
        "board": [str(c) for c in getattr(state, "community", [])],
        "pot": getattr(state, "pot", 0),
        "current_player": getattr(state, "current_player", 0),
        "legal_actions": [str(a) for a in getattr(state, "legal_actions", [])],
        "final_state": getattr(state, "final_state", False),
        "winner": [],
        "players": []
    }

    for i, p in enumerate(state.players_state):
        data["players"].append({
            "id": i,
            "stack": getattr(p, "stack", 0),
            "bet": getattr(p, "bet", 0),
            "active": getattr(p, "active", False),
            "hand": [str(c) for c in getattr(p, "hand", [])] if i == 0 else ["🂠", "🂠"]
        })

    if getattr(state, "final_state", False):
        data["winner"] = [
            i for i, p in enumerate(state.players_state)
            if getattr(p, "reward", 0) > 0
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
