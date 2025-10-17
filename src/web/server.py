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
            "stack": getattr(p, "stake", 0),
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
    print("=== DEBUG JSON ===")
    import json
    print(json.dumps(data, indent=2, ensure_ascii=False))

    return data





# ---------- 路由 ----------
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/start", methods=["POST"])
def start():
    """
    启动一局新游戏：
    1. 初始化状态
    2. 自动让 AI 玩家执行动作直到轮到玩家 0
    3. 返回可直接渲染的 JSON 状态
    """
    global CURRENT_STATE
    import random
    print("\n================= [DEBUG] /start 被调用 =================")

    try:
        print("[1] 准备调用 State.from_seed() ...")
        CURRENT_STATE = State.from_seed(
            n_players=6,
            sb=1,
            bb=2,
            button=random.randint(0, 5),  # 随机庄位，更真实
            stake=200.0,
            seed=random.randint(0, 100000)
        )
        print("[2] State.from_seed() 返回成功")
    except Exception as e:
        print("❌ [ERROR] State.from_seed 抛出异常：", e)
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"State.from_seed failed: {str(e)}"}), 500

    print(f"[3] 当前玩家: {CURRENT_STATE.current_player}")
    print("=== DEBUG HANDS ===")
    for i, p in enumerate(CURRENT_STATE.players_state):
        print(f"Player {i} hand:", p.hand)
    print("===================")

    # 让 AI 玩家自动行动直到轮到玩家 0 或游戏结束
    try:
        step = 0
        print("[4] 开始让 AI 自动执行动作 ...")
        while (
            not CURRENT_STATE.final_state
            and CURRENT_STATE.current_player != 0
            and len(CURRENT_STATE.legal_actions) > 0
            and step < 50
        ):
            ai_action = AI_AGENT.choose_action(CURRENT_STATE)
            print(f"🤖 AI 玩家 {CURRENT_STATE.current_player} 执行动作: {ai_action}")
            new_state = CURRENT_STATE.apply_action(ai_action)

            if new_state.status != StateStatus.Ok:
                print(f"⚠️ 非法动作: {new_state.status}")
                break

            CURRENT_STATE = new_state
            step += 1

        print(f"✅ AI 执行 {step} 步后，轮到玩家 {CURRENT_STATE.current_player}")
    except Exception as e:
        print("❌ [ERROR] AI 自动动作出错：", e)
        import traceback
        traceback.print_exc()

    # 输出当前状态
    print("[5] 准备 serialize_state()")
    try:
        data = serialize_state(CURRENT_STATE)
        print("[6] serialize_state() 成功，准备返回 JSON")
        return jsonify(data)
    except Exception as e:
        print("❌ [ERROR] serialize_state 出错：", e)
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"serialize_state failed: {str(e)}"}), 500




from pokers import Action, ActionEnum

@app.route("/act", methods=["POST"])
def act():
    global CURRENT_STATE

    if not CURRENT_STATE:
        return jsonify({"error": "Game not started"}), 400

    data = request.get_json()
    action_id = data.get("action_id", 0)
    legal = CURRENT_STATE.legal_actions

    if not legal:
        return jsonify({"error": "No legal actions"}), 400

    try:
        selected_enum = legal[action_id]
    except IndexError:
        selected_enum = legal[0]

    print(f"你执行动作: {selected_enum}")

    # ✅ 玩家动作
    if isinstance(selected_enum, ActionEnum):
        if selected_enum == ActionEnum.Raise:
            player_action = Action(ActionEnum.Raise, amount=10.0)
        else:
            player_action = Action(selected_enum)
    elif isinstance(selected_enum, Action):
        player_action = selected_enum
    else:
        return jsonify({"error": f"未知动作类型: {type(selected_enum)}"}), 400

    CURRENT_STATE = CURRENT_STATE.apply_action(player_action)

    # ✅ AI 连续执行
    step = 0
    while (
        not CURRENT_STATE.final_state
        and CURRENT_STATE.current_player != 0
        and len(CURRENT_STATE.legal_actions) > 0
        and step < 50
    ):
        ai_action_enum = AI_AGENT.choose_action(CURRENT_STATE)
        print(f"🤖 AI 玩家 {CURRENT_STATE.current_player} 执行动作: {ai_action_enum}")

        # ✅ 判断返回类型，兼容 ActionEnum 或 Action
        if isinstance(ai_action_enum, ActionEnum):
            if ai_action_enum == ActionEnum.Raise:
                ai_action = Action(ActionEnum.Raise, amount=10.0)
            else:
                ai_action = Action(ai_action_enum)
        elif isinstance(ai_action_enum, Action):
            ai_action = ai_action_enum
        else:
            print(f"⚠️ AI 返回未知类型: {type(ai_action_enum)}，跳过")
            break

        new_state = CURRENT_STATE.apply_action(ai_action)
        if new_state.status != StateStatus.Ok:
            print(f"⚠️ AI 非法动作: {new_state.status}")
            break

        CURRENT_STATE = new_state
        step += 1

    print(f"✅ AI 执行 {step} 步后，轮到玩家 {CURRENT_STATE.current_player}")
    return jsonify(serialize_state(CURRENT_STATE))





if __name__ == "__main__":
    port = 5000
    print(f"✅ Poker Web 服务器启动中：http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
