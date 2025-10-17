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
    """把 poker 状态对象转换成可前端显示的 JSON 格式"""

    def card_to_str(card):
        """把 pokers.Card 转换为可视字符"""
        try:
            # 有些版本 pokers.Card 没公开 rank/suit，可以转成字符串或访问属性
            if hasattr(card, "rank") and hasattr(card, "suit"):
                rank = str(card.rank)
                suit_idx = card.suit
            else:
                # 兼容 fallback：转成字符串（例如 'AS', 'QH'）
                s = str(card)
                rank = s[0].upper()
                suit_char = s[-1].lower()
                suits = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
                suit = suits.get(suit_char, '?')
                return f"{rank}{suit}"

            ranks = {
                "2": "2", "3": "3", "4": "4", "5": "5", "6": "6",
                "7": "7", "8": "8", "9": "9", "10": "10",
                "11": "J", "12": "Q", "13": "K", "14": "A"
            }
            suits_map = ["♣", "♦", "♥", "♠"]
            rank_str = ranks.get(str(rank), str(rank))
            suit_str = suits_map[int(suit_idx) % 4]
            return f"{rank_str}{suit_str}"
        except Exception as e:
            print("⚠️ card_to_str error:", e)
            return "??"

    players = []
    for i, p in enumerate(state.players_state):
        hand_strs = [card_to_str(c) for c in getattr(p, "hand", [])]
        players.append({
            "id": i,
            "name": f"Player {i}",
            "stack": getattr(p, "stack", 0),
            "hand": hand_strs
        })

    community_cards = []
    try:
        community_cards = [card_to_str(c) for c in getattr(state, "community", [])]
    except Exception:
        pass

    legal_actions = []
    try:
        legal_actions = [str(a) for a in getattr(state, "legal_actions", [])]
    except Exception:
        pass

    pot_value = getattr(state, "pot", 0)
    if hasattr(pot_value, "value"):
        pot_value = pot_value.value

    data = {
        "pot": pot_value,
        "players": players,
        "community": community_cards,
        "legal_actions": legal_actions,
        "current_player": getattr(state, "current_player", -1),
        "final_state": getattr(state, "final_state", False)
    }

    # ✅ 打印一次转换后的 JSON（不递归调用自己）

    return data





# ---------- 路由 ----------
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/start", methods=["POST"])
def start():
    global CURRENT_STATE
    import random
    CURRENT_STATE = State.from_seed(
        n_players=6,
        sb=1,
        bb=2,
        button=0,
        stake=200.0,
        seed=random.randint(0, 100000)
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
    legal = getattr(CURRENT_STATE, "legal_actions", [])
    if not legal:
        return jsonify({"error": "No legal actions"}), 400

    try:
        player_action = legal[action_id]
    except IndexError:
        player_action = legal[0]

    print(f"🧍‍♂️ 你执行动作: {player_action}")
    CURRENT_STATE = CURRENT_STATE.apply_action(player_action)

    # === 让 AI 自动执行多轮（安全循环） ===
    max_steps = 30
    step = 0

    while (
        not CURRENT_STATE.final_state
        and CURRENT_STATE.current_player != 0
        and len(CURRENT_STATE.legal_actions) > 0
        and step < max_steps
    ):
        ai_action = AI_AGENT.choose_action(CURRENT_STATE)
        print(f"🤖 AI 玩家 {CURRENT_STATE.current_player} 执行动作: {ai_action}")
        new_state = CURRENT_STATE.apply_action(ai_action)

        if new_state.status != StateStatus.Ok:
            print(f"⚠️ 非法动作：{new_state.status}")
            break

        CURRENT_STATE = new_state
        step += 1

    return jsonify(serialize_state(CURRENT_STATE))



if __name__ == "__main__":
    port = 5000
    print(f"✅ Poker Web 服务器启动中：http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
