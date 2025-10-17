import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from flask import Flask, jsonify, request, send_from_directory
import torch
import pokers as pkrs
from src.core.deep_cfr import DeepCFRAgent
from src.agents.random_agent import RandomAgent
from src.utils.settings import set_strict_checking

set_strict_checking(False)

app = Flask(__name__, static_folder="static")

GAME_STATE = None
AI_AGENT = None

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/start", methods=["POST"])
def start():
    global GAME_STATE, AI_AGENT
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load("models/checkpoint_iter_100.pt", map_location=device)
    AI_AGENT = DeepCFRAgent(player_id=1, num_players=2, device=device)
    AI_AGENT.advantage_net.load_state_dict(ckpt["advantage_net"])
    AI_AGENT.strategy_net.load_state_dict(ckpt["strategy_net"])

    GAME_STATE = pkrs.State.from_seed(n_players=2, sb=1, bb=2, stake=200.0, seed=42)
    return jsonify(serialize_state(GAME_STATE))

@app.route("/act", methods=["POST"])
def act():
    global GAME_STATE
    if GAME_STATE is None:
        return jsonify({"error": "Game not started"})

    action_id = request.json.get("action_id")
    legal_actions = GAME_STATE.get_legal_actions()
    if not (0 <= action_id < len(legal_actions)):
        return jsonify({"error": "Invalid action"})

    # 玩家动作
    GAME_STATE = GAME_STATE.apply_action(legal_actions[action_id])
    if GAME_STATE.status != pkrs.StateStatus.Ok:
        return jsonify({"error": f"Illegal state: {GAME_STATE.status}"})

    # AI 自动行动
    while not GAME_STATE.final_state and GAME_STATE.current_player == 1:
        ai_action = AI_AGENT.choose_action(GAME_STATE)
        GAME_STATE = GAME_STATE.apply_action(ai_action)
        if GAME_STATE.status != pkrs.StateStatus.Ok:
            return jsonify({"error": f"AI illegal state: {GAME_STATE.status}"})

    return jsonify(serialize_state(GAME_STATE))

def serialize_state(state):
    return {
        "stage": state.stage.name,
        "pot": state.pot,
        "current_player": state.current_player,
        "final_state": state.final_state,
        "board": [str(c) for c in state.board],
        "players": [
            {
                "id": i,
                "stack": ps.stack,
                "bet": ps.bet,
                "folded": ps.folded,
                "reward": ps.reward,
            }
            for i, ps in enumerate(state.players_state)
        ],
        "legal_actions": [str(a) for a in state.get_legal_actions()] if not state.final_state and state.current_player == 0 else [],
        "winner": [i for i, ps in enumerate(state.players_state) if ps.reward > 0] if state.final_state else [],
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
