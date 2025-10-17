# src/web/server.py
from flask import Flask, jsonify, request, send_from_directory
import torch
import random
from pokers import State, StateStatus, Action, ActionEnum
from src.core.deep_cfr import DeepCFRAgent
from src.agents.random_agent import RandomAgent

app = Flask(__name__, static_folder="static", static_url_path="/static")

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "src/training/checkpoint_iter_100.pt"


# ---------- 加载AI ----------
def safe_load_agent():
    """加载AI模型，如果失败则使用随机AI"""
    try:
        print(f"🔹 正在加载AI模型：{MODEL_PATH}")
        agent = DeepCFRAgent(player_id=0, num_players=6, device=device)
        ckpt = torch.load(MODEL_PATH, map_location=device)
        agent.advantage_net.load_state_dict(ckpt["advantage_net"], strict=False)
        agent.strategy_net.load_state_dict(ckpt["strategy_net"], strict=False)
        print("✅ 模型加载成功")
        return agent
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("⚠️ 启用随机AI替代")
        return RandomAgent(0)


AI_AGENT = safe_load_agent()
CURRENT_STATE = None


# ---------- 状态转JSON ----------
def serialize_state(state):
    """将State对象转换为前端可用的JSON并打印牌信息（带摊牌/隐藏逻辑）"""
    print("\n=== [DEBUG] serialize_state() 调用 ===")

    # ---------- 工具：Card -> 文本 ----------
    def card_to_str(card):
        """智能转换 pokers.Card → 扑克牌符号字符串"""
        try:
            if hasattr(card, "rank") and hasattr(card, "suit"):
                rank_val = int(card.rank)
                suit_val = int(card.suit)
            else:
                # 退化路径：字符串形式 'Ah','Td' 等
                s = str(card)
                # 尝试解析 'Ah' 这类
                suits_map_rev = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
                if len(s) >= 2 and s[-1].lower() in suits_map_rev:
                    rank_str = s[:-1].upper().replace("10", "T")
                    suit_str = suits_map_rev[s[-1].lower()]
                    # Rank 归一：T/J/Q/K/A 其余数字原样
                    rank_map = {"T": "10", "J": "J", "Q": "Q", "K": "K", "A": "A"}
                    if rank_str in rank_map:
                        return f"{rank_map[rank_str]}{suit_str}"
                    return f"{rank_str}{suit_str}"
                # 兜底：原样返回，前端 normalizeCard 再修
                print(f"⚠️ 未标准Card对象: {s}")
                return s

            # pokers 有的实现 rank 是 0-12，这里统一偏移到 2-A
            if 0 <= rank_val <= 12:
                rank_val += 2

            ranks_map = {
                2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
                7: "7", 8: "8", 9: "9", 10: "10",
                11: "J", 12: "Q", 13: "K", 14: "A"
            }
            suits_map = {0: "♣", 1: "♦", 2: "♥", 3: "♠"}

            rank_str = ranks_map.get(rank_val, str(rank_val))
            suit_str = suits_map.get(suit_val % 4, "?")
            return f"{rank_str}{suit_str}"
        except Exception as e:
            print(f"⚠️ card_to_str 出错: {e} ({card})")
            return "??"

    # ---------- 阶段/结束态 ----------
    final_state = bool(getattr(state, "final_state", False))
    stage_val = getattr(state, "stage", "")
    print(f"[阶段] {stage_val} | final_state={final_state}")

    # ---------- 公共牌 ----------
    community_cards = []
    for attr in ("public_cards", "community", "board"):
        if hasattr(state, attr):
            cards = getattr(state, attr)
            if cards:
                community_cards = [card_to_str(c) for c in cards]
                break
    print(f"[公共牌] {community_cards if community_cards else '[]'}")

    # ---------- 玩家 ----------
    players = []
    for i, p in enumerate(state.players_state):
        # 原始手牌
        raw_hand_cards = getattr(p, "hand", [])
        hand_str_all = [card_to_str(c) for c in raw_hand_cards]

        active = bool(getattr(p, "active", True))
        stake_val = float(getattr(p, "stake", getattr(p, "stack", 0.0)))

        # 显示逻辑：
        # - 进行中：仅你（i==0 且 active=True）显示；AI 与 fold 都隐藏
        # - 结束：仅 active=True 的玩家显示真实牌；fold 的一直不显示
        if not final_state:
            if i == 0 and active:
                display_hand = hand_str_all
            else:
                display_hand = []  # 隐藏
        else:
            display_hand = hand_str_all if active else []

        # 调试日志
        print(f"玩家 {i} | active={active} | stake={stake_val:.2f} | "
              f"手牌(原始)={hand_str_all} | 手牌(展示)={display_hand}")

        players.append({
            "id": i,
            "name": f"Player {i}",
            "stack": stake_val,
            "hand": display_hand,
            "active": active
        })

    # ---------- 合法动作（简名） ----------
    legal_actions = []
    try:
        for a in getattr(state, "legal_actions", []):
            if hasattr(a, "name"):
                legal_actions.append(a.name)         # Fold/Call/Check/Raise
            else:
                legal_actions.append(str(a).split(".")[-1])  # 兜底去掉前缀
    except Exception as e:
        print("⚠️ legal_actions 解析异常:", e)

    # ---------- 底池 ----------
    pot_value = getattr(state, "pot", 0.0)
    if hasattr(pot_value, "value"):
        pot_value = float(pot_value.value)
    else:
        pot_value = float(pot_value)

    # ---------- 赢家（仅结束时携带） ----------
    winners = []
    rewards = []
    if final_state:
        rewards = [float(getattr(p, "reward", 0.0)) for p in state.players_state]
        if rewards:
            max_r = max(rewards)
            winners = [i for i, r in enumerate(rewards) if r == max_r]
        print(f"[结算] rewards={rewards} | winners={winners}")

    # ---------- 打包结果 ----------
    data = {
        "pot": pot_value,
        "players": players,
        "community": community_cards,
        "legal_actions": legal_actions,
        "current_player": int(getattr(state, "current_player", -1)),
        "final_state": final_state,
        "stage": str(stage_val),
    }
    if final_state:
        data["winners"] = winners
        data["rewards"] = rewards

    # ---------- 打印完整JSON ----------
    import json
    print("=== [DEBUG JSON 输出] ===")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print("=== [DEBUG JSON 结束] ===")

    return data



# ---------- 路由 ----------
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/start", methods=["POST"])
def start():
    """开启新的一局"""
    global CURRENT_STATE
    print("\n================= [DEBUG] /start =================")
    try:
        CURRENT_STATE = State.from_seed(
            n_players=6,
            sb=1,
            bb=2,
            button=random.randint(0, 5),
            stake=200.0,
            seed=random.randint(0, 99999)
        )
        print("✅ State 初始化成功")
    except Exception as e:
        print("❌ State 初始化失败:", e)
        return jsonify({"error": str(e)}), 500

    # AI 自动执行直到轮到玩家 0 或游戏结束
    step = 0
    while not CURRENT_STATE.final_state:
        if CURRENT_STATE.current_player == 0:
            break
        ai_action = AI_AGENT.choose_action(CURRENT_STATE)
        CURRENT_STATE = CURRENT_STATE.apply_action(ai_action)
        step += 1

    print(f"✅ AI 执行 {step} 步后，轮到玩家 {CURRENT_STATE.current_player}")
    return jsonify(serialize_state(CURRENT_STATE))


@app.route("/act", methods=["POST"])
def act():
    """玩家执行动作"""
    global CURRENT_STATE
    if not CURRENT_STATE:
        return jsonify({"error": "Game not started"}), 400

    data = request.get_json()
    action_id = data.get("action_id", 0)
    legal = CURRENT_STATE.legal_actions
    if not legal:
        return jsonify({"error": "No legal actions"}), 400

    # 解析玩家动作
    try:
        selected_enum = legal[action_id]
    except IndexError:
        selected_enum = legal[0]
    print(f"🧍‍♂️ 玩家执行动作: {selected_enum}")

    # 构造 Action 对象
    if isinstance(selected_enum, ActionEnum):
        if selected_enum == ActionEnum.Raise:
            player_action = Action(ActionEnum.Raise, amount=10.0)
        else:
            player_action = Action(selected_enum)
    else:
        player_action = selected_enum

    CURRENT_STATE = CURRENT_STATE.apply_action(player_action)

    # 让 AI 执行直到玩家 0 或游戏结束
    step = 0
    while not CURRENT_STATE.final_state:
        if CURRENT_STATE.current_player == 0:
            break
        ai_action = AI_AGENT.choose_action(CURRENT_STATE)
        CURRENT_STATE = CURRENT_STATE.apply_action(ai_action)
        step += 1

    print(f"✅ AI 执行 {step} 步后，轮到玩家 {CURRENT_STATE.current_player}")
    return jsonify(serialize_state(CURRENT_STATE))


if __name__ == "__main__":
    port = 5000
    print(f"✅ Poker Web 服务器启动：http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
