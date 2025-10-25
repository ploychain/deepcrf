# src/web/server.py
import os
from glob import glob

from flask import Flask, jsonify, request, send_from_directory
import torch
import random
from pokers import State, StateStatus, Action, ActionEnum
from src.core.deep_cfr import DeepCFRAgent
from src.agents.random_agent import RandomAgent

app = Flask(__name__, static_folder="static", static_url_path="/static")

device = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL_PATH = "models/checkpoint_iter_1000.pt"


# ---------- 加载AI ----------
def discover_model_paths():
    """扫描常见目录获取可用模型路径"""
    candidates = []
    search_dirs = [
        "fix_models2",
        # "models/checkpoints",
        # "flagship_models",
        # "flagship_models/first",
    ]
    for path in search_dirs:
        if not os.path.isdir(path):
            continue
        candidates.extend(sorted(glob(os.path.join(path, "*.pt"))))

    if not candidates and os.path.isfile(DEFAULT_MODEL_PATH):
        candidates.append(DEFAULT_MODEL_PATH)

    return candidates


def safe_load_agent(player_id, model_path):
    """加载指定座位与模型文件的AI，如果失败则使用随机AI"""
    if not model_path:
        print(f"⚠️ 玩家 {player_id} 未提供模型路径，改用随机AI")
        return RandomAgent(player_id)

    try:
        print(f"🔹 正在为玩家 {player_id} 加载AI模型：{model_path}")
        agent = DeepCFRAgent(player_id=player_id, num_players=6, device=device)
        agent.load_model(model_path)
        print(f"✅ 模型加载成功（玩家 {player_id}）")
        return agent
    except Exception as e:
        print(f"❌ 玩家 {player_id} 模型加载失败: {e}")
        print(f"⚠️ 启用随机AI替代（玩家 {player_id}）")
        return RandomAgent(player_id)


MODEL_PATHS = discover_model_paths()
if MODEL_PATHS:
    print("🔍 检测到以下模型文件用于AI对手：")
    for idx, path in enumerate(MODEL_PATHS):
        print(f"  [{idx}] {path}")
else:
    print("⚠️ 未发现任何模型文件，将全部使用随机AI")

# 玩家 0 是用户，其余位置为 AI；若模型不足则循环使用
AI_AGENTS = [None]
for pid in range(1, 6):
    model_path = MODEL_PATHS[(pid - 1) % len(MODEL_PATHS)] if MODEL_PATHS else None
    AI_AGENTS.append(safe_load_agent(pid, model_path))
CURRENT_STATE = None


# ---------- 状态转JSON ----------
def serialize_state(state):
    """将State对象转换为前端可用的JSON并打印牌信息"""
    print("\n=== [DEBUG] serialize_state() 调用 ===")

    num_players = len(getattr(state, "players_state", []))
    button_pos = getattr(state, "button", -1)
    sb_pos = (button_pos + 1) % num_players if num_players else -1
    bb_pos = (button_pos + 2) % num_players if num_players > 1 else -1

    stage_obj = getattr(state, "stage", "")
    stage_name = getattr(stage_obj, "name", str(stage_obj))
    try:
        stage_index = int(stage_obj)
    except Exception:
        stage_index = None

    def safe_float(val, default=0.0):
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def card_to_str(card):
        """智能转换 pokers.Card → 扑克牌符号字符串"""
        try:
            if hasattr(card, "rank") and hasattr(card, "suit"):
                rank_val = int(card.rank)
                suit_val = int(card.suit)
            else:
                s = str(card)
                print(f"⚠️ 未标准Card对象: {s}")
                return s

            # 动态修正 rank：如果是 0–12，则映射到 2–A
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

    def action_to_str(action):
        """转换玩家动作为易读文本"""
        if not action:
            return ""
        try:
            act_obj = getattr(action, "action", action)
            act_name = getattr(act_obj, "name", str(act_obj))
            amount = getattr(action, "amount", None)
            if amount is None and hasattr(action, "bet"):
                amount = getattr(action, "bet", None)
            if isinstance(amount, (int, float)) and amount > 0:
                return f"{act_name} {amount:.2f}"
            return act_name
        except Exception as e:
            print(f"⚠️ action_to_str 出错: {e} ({action})")
            return str(action)

    # ---------- 玩家 ----------
    players = []
    for i, p in enumerate(state.players_state):
        hand_cards = getattr(p, "hand", [])
        hand_str = [card_to_str(c) for c in hand_cards]
        active = bool(getattr(p, "active", True))
        action = getattr(p, "legal_actions", "")
        bet_amount = safe_float(getattr(p, "bet_chips", 0.0))
        reward = safe_float(getattr(p, "reward", 0.0))
        last_action = action_to_str(getattr(p, "last_action", ""))
        players.append({
            "id": i,
            "name": f"Player {i}",
            "stack": safe_float(getattr(p, "stake", 0)),
            "hand": hand_str,
            "active": active,
            "action": action,
            "bet": bet_amount,
            "reward": reward,
            "last_action": last_action,
            "is_dealer": i == button_pos,
            "is_small_blind": i == sb_pos,
            "is_big_blind": i == bb_pos,
            "position_index": (i - button_pos) % num_players if num_players else None,
        })
        print(f"玩家 {i} 手牌: {hand_str}")

    # ---------- 公共牌 ----------
    community_cards = []
    for attr in ("community", "board", "public_cards"):
        if hasattr(state, attr):
            cards = getattr(state, attr)
            if cards:
                community_cards = [card_to_str(c) for c in cards]
                break

    print(f"公共牌: {community_cards if community_cards else '[]'}")

    # ---------- 合法动作 ----------
    legal_actions = [str(a) for a in getattr(state, "legal_actions", [])]

    # ---------- 底池 ----------
    pot_value = getattr(state, "pot", 0)
    if hasattr(pot_value, "value"):
        pot_value = pot_value.value
    pot_value = safe_float(pot_value, 0.0)

    # ---------- 判断赢家 ----------
    winner_ids = []
    if getattr(state, "final_state", False):
        rewards = [safe_float(getattr(p, "reward", 0), 0.0) for p in state.players_state]
        max_reward = max(rewards)
        if max_reward > 0:
            winner_ids = [i for i, r in enumerate(rewards) if r == max_reward]
        print(f"🏆 检测到赢家: {winner_ids}, 奖励分布: {rewards}")

    sb_amount = getattr(state, "sb", None)
    bb_amount = getattr(state, "bb", None)
    if sb_amount is None:
        sb_amount = getattr(state, "small_blind", 0)
    if bb_amount is None:
        bb_amount = getattr(state, "big_blind", 0)
    sb_amount = safe_float(sb_amount, 0.0)
    bb_amount = safe_float(bb_amount, 0.0)

    data = {
        "pot": pot_value,
        "players": players,
        "community": community_cards,
        "legal_actions": legal_actions,
        "current_player": getattr(state, "current_player", -1),
        "final_state": getattr(state, "final_state", False),
        "stage": stage_name,
        "stage_index": stage_index,
        "winner": winner_ids,
        "button": button_pos,
        "small_blind_player": sb_pos,
        "big_blind_player": bb_pos,
        "small_blind_amount": sb_amount,
        "big_blind_amount": bb_amount,
    }

    # ---------- 打印完整JSON ----------
    import json
    print("=== [DEBUG JSON 输出] ===")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print("=== [DEBUG JSON 结束] ===")

    return data


def describe_action(action):
    """格式化动作信息"""
    try:
        act_enum = getattr(action, "action", action)
        act_name = getattr(act_enum, "name", str(act_enum))
        amount = getattr(action, "amount", None)
        if amount is None and hasattr(action, "bet"):
            amount = getattr(action, "bet", None)
        if isinstance(amount, (int, float)) and amount > 0:
            return f"{act_name}({amount:.2f})"
        return act_name
    except Exception:
        return str(action)


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
        current_seat = CURRENT_STATE.current_player
        agent = AI_AGENTS[current_seat]
        if agent is None:
            print(f"⚠️ 未找到玩家 {current_seat} 的 AI，停止自动行动")
            break
        legal_desc = ", ".join(describe_action(a) for a in CURRENT_STATE.legal_actions)
        print(f"🤖 玩家 {current_seat} 合法动作: [{legal_desc}]")
        ai_action = agent.choose_action(CURRENT_STATE)
        print(f"🤖 玩家 {current_seat} 选择: {describe_action(ai_action)}")
        # 🔍 调试下注输出
        if hasattr(ai_action, "amount"):
            print(f"🤖 下注预测值: {ai_action.amount}")
        else:
            print("🤖 下注预测值: 无 amount 字段")

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
        current_seat = CURRENT_STATE.current_player
        agent = AI_AGENTS[current_seat]
        if agent is None:
            print(f"⚠️ 未找到玩家 {current_seat} 的 AI，停止自动行动")
            break
        legal_desc = ", ".join(describe_action(a) for a in CURRENT_STATE.legal_actions)
        print(f"🤖 玩家 {current_seat} 合法动作: [{legal_desc}]")
        ai_action = agent.choose_action(CURRENT_STATE)
        print(f"🤖 玩家 {current_seat} 选择: {describe_action(ai_action)}")
        CURRENT_STATE = CURRENT_STATE.apply_action(ai_action)
        step += 1

    print(f"✅ AI 执行 {step} 步后，轮到玩家 {CURRENT_STATE.current_player}")
    return jsonify(serialize_state(CURRENT_STATE))


if __name__ == "__main__":
    import os
    from werkzeug.serving import run_simple

    port = int(os.environ.get("PORT", 5000))
    print(f"✅ Poker Web 服务器启动：http://0.0.0.0:{port}")
    run_simple("0.0.0.0", port, app, use_reloader=False, use_debugger=False)

