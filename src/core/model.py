# coding: utf-8
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pokers as pkrs
from src.core.hand_straighty_potential import hand_straighty_potential
from src.core.hand_flushy_potential import hand_flushy_potential
from src.core.highcard_on_board_norm import highcard_on_board_norm
from src.core.lowcard_on_board_norm import lowcard_on_board_norm
from src.core.board_gap_norm import board_gap_norm
from src.core.avg_rank_on_board_norm import avg_rank_on_board_norm
from src.core.paired_level import paired_level
from src.core.straighty_hint import straighty_hint
from src.core.flushy_hint import flushy_hint
from src.core.flushy_hint import *



# === 可选：treys 用于 MC 兜底（不存在也不影响） ===
try:
    from treys import Card as TCard, Deck as TDeck, Evaluator as TEvaluator
    _TREYS_OK = True
    _teval = TEvaluator()
except Exception:
    _TREYS_OK = False
    _teval = None

# ===================== 全局常量与配置 =====================
VERBOSE = False
# === 按需修改你的 CSV 路径 ===
EQUITY_PATH   = '/home/harry/deepcfr/equity_table.csv'
FLOP_CSV_PATH = '/home/harry/deepcfr/flop_strength_table_nsim500.csv'
TURN_CSV_PATH = '/home/harry/deepcfr/turn_strength_table_nsim500.csv'

# 预加载 Equity（preflop）表
equity_table = pd.read_csv(EQUITY_PATH, dtype={'hand': str, 'equity': float})
equity_table['hand'] = equity_table['hand'].astype(str).str.strip().str.upper()

# 52 one-hot 索引需要的映射（适配 pokers 的 rank/suit 字符串）
RANK_MAP = {'R2': 1, 'R3': 2, 'R4': 3, 'R5': 4, 'R6': 5, 'R7': 6,
            'R8': 7, 'R9': 8, 'RT': 9, 'RJ': 10, 'RQ': 11, 'RK': 12, 'RA': 13}
SUIT_MAP = {'Spades': 0, 'Hearts': 1, 'Diamonds': 2, 'Clubs': 3}

# hand_type/canonical 使用的字符映射
RANK_CHAR_MAP = {'R2': '2', 'R3': '3', 'R4': '4', 'R5': '5', 'R6': '6', 'R7': '7',
                 'R8': '8', 'R9': '9', 'RT': 'T', 'RJ': 'J', 'RQ': 'Q', 'RK': 'K', 'RA': 'A'}
SUIT_CHAR_MAP = {'Spades': 's', 'Hearts': 'h', 'Diamonds': 'd', 'Clubs': 'c'}
_RANKS_ORDER = "23456789TJQKA"  # 2..A

def set_verbose(verbose_mode: bool):
    global VERBOSE
    VERBOSE = verbose_mode

# =============== 基础工具 ===============
def encode_cards(cards):
    """任意牌面列表 -> 52 维 one-hot（与你项目一致的索引方案）"""
    vec = np.zeros(52, dtype=np.float32)
    for card in cards:
        rank_str = str(card.rank).split('.')[-1]
        suit_str = str(card.suit).split('.')[-1]
        idx = SUIT_MAP[suit_str] * 13 + RANK_MAP[rank_str]
        vec[idx] = 1.0
    return vec

def get_preflop_equity(hand_cards):
    """两张手牌 -> preflop equity（查 equity_table）"""
    if not hand_cards:
        return 0.0
    ranks = [str(c.rank).split('.')[-1] for c in hand_cards]
    suits = [str(c.suit).split('.')[-1] for c in hand_cards]
    rank_order = list(RANK_MAP.keys())
    sorted_cards = sorted(zip(ranks, suits), key=lambda x: rank_order.index(x[0]), reverse=True)
    r1, s1 = sorted_cards[0]
    r2, s2 = sorted_cards[1]
    h1, h2 = RANK_CHAR_MAP[r1], RANK_CHAR_MAP[r2]
    suited = 's' if s1 == s2 else 'o'
    hand_str = f"{h1}{h2}{suited}".upper()
    row = equity_table.loc[equity_table['hand'] == hand_str]
    if not row.empty:
        eq = float(row['equity'].values[0])
        if VERBOSE:
            print(f"[DEBUG] preflop equity {hand_str} = {eq:.6f}")
        return eq
    if VERBOSE:
        print(f"[WARN] preflop hand not found: {hand_str}")
    return 0.0

# =============== LUT（flop/turn）加载与规范化 ===============
def _norm_canon_lower(canon_text: str) -> str:
    """把 'ranks|suits' 的 suits 段统一为小写，并去空白"""
    ct = str(canon_text).strip()
    if '|' in ct:
        ranks, suits = ct.split('|', 1)
        return ranks + '|' + suits.lower()
    return ct.lower()

def _load_lut_csv(path, street='flop'):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        if VERBOSE:
            print(f"[WARN] read {street} csv failed: {e}")
        return {}
    if street == 'flop':
        df["hand_type"]      = df["hand_type"].astype(str).str.strip().str.upper()
        df["flop_pattern"]   = df["flop_pattern"].astype(str).str.strip().str.lower()
        df["suit_signature"] = df["suit_signature"].astype(str).str.strip().str.lower()
        df["canon_flop"]     = df["canon_flop"].apply(_norm_canon_lower)
        key_cols = ("hand_type","canon_flop","flop_pattern","suit_signature")
    else:
        df["hand_type"]      = df["hand_type"].astype(str).str.strip().str.upper()
        df["turn_pattern"]   = df["turn_pattern"].astype(str).str.strip().str.lower()
        df["suit_signature"] = df["suit_signature"].astype(str).str.strip().str.lower()
        df["canon_turn"]     = df["canon_turn"].apply(_norm_canon_lower)
        key_cols = ("hand_type","canon_turn","turn_pattern","suit_signature")
    lut = {}
    for _, r in df.iterrows():
        key = tuple(r[k] for k in key_cols)
        lut[key] = float(r["strength_mean"])
    return lut

_FLOP_LUT = None
_TURN_LUT = None
def _get_flop_lut():
    global _FLOP_LUT
    if _FLOP_LUT is None:
        _FLOP_LUT = _load_lut_csv(FLOP_CSV_PATH, 'flop')
    return _FLOP_LUT

def _get_turn_lut():
    global _TURN_LUT
    if _TURN_LUT is None:
        _TURN_LUT = _load_lut_csv(TURN_CSV_PATH, 'turn')
    return _TURN_LUT

# =============== hand_type / canonical / suit-signature ===============
def _hand_type_169_from_state_cards(c1, c2) -> str:
    r1 = RANK_CHAR_MAP[str(c1.rank).split('.')[-1]]
    r2 = RANK_CHAR_MAP[str(c2.rank).split('.')[-1]]
    s1 = SUIT_CHAR_MAP[str(c1.suit).split('.')[-1]]
    s2 = SUIT_CHAR_MAP[str(c2.suit).split('.')[-1]]
    if _RANKS_ORDER.index(r1) < _RANKS_ORDER.index(r2):
        r1, r2, s1, s2 = r2, r1, s2, s1
    if r1 == r2:
        return f"{r1}{r2}"
    return f"{r1}{r2}{'s' if s1 == s2 else 'o'}".upper()

def _canonicalize_flop_from_state(board3):
    triples = []
    for c in board3:
        rch = RANK_CHAR_MAP[str(c.rank).split('.')[-1]]
        sch = SUIT_CHAR_MAP[str(c.suit).split('.')[-1]]
        rint = _RANKS_ORDER.index(rch) + 2
        triples.append((rint, sch, c))
    triples.sort(key=lambda t: (-t[0], t[1]))
    ranks_sorted = [t[0] for t in triples]
    suit_seen, next_label, labels = {}, 0, []
    for _, s, _ in triples:
        if s not in suit_seen:
            suit_seen[s] = next_label; next_label += 1
        labels.append(chr(ord('a') + suit_seen[s]))
    canon = f"{','.join(map(str, ranks_sorted))}|{''.join(labels)}"
    pat = ''.join(labels)
    return canon, pat, suit_seen  # suit_seen: {'s':0, 'h':1 ...}

def _canonicalize_turn_from_state(board4):
    quads = []
    for c in board4:
        rch = RANK_CHAR_MAP[str(c.rank).split('.')[-1]]
        sch = SUIT_CHAR_MAP[str(c.suit).split('.')[-1]]
        rint = _RANKS_ORDER.index(rch) + 2
        quads.append((rint, sch, c))
    quads.sort(key=lambda t: (-t[0], t[1]))
    ranks_sorted = [t[0] for t in quads]
    suit_seen, next_label, labels = {}, 0, []
    for _, s, _ in quads:
        if s not in suit_seen:
            suit_seen[s] = next_label; next_label += 1
        labels.append(chr(ord('a') + suit_seen[s]))
    canon = f"{','.join(map(str, ranks_sorted))}|{''.join(labels)}"
    pat = ''.join(labels)
    return canon, pat, suit_seen

def _suit_sig_vs_seen_for_hole(hole2, suit_seen):
    def rch(card): return RANK_CHAR_MAP[str(card.rank).split('.')[-1]]
    c_high, c_low = sorted(hole2, key=lambda c: _RANKS_ORDER.index(rch(c)), reverse=True)
    inv = {idx: set() for idx in suit_seen.values()}
    for sch, idx in suit_seen.items():
        inv[idx].add(sch)
    label_to_suits = {chr(ord('a') + k): v for k, v in inv.items()}
    def tok(card):
        sch = SUIT_CHAR_MAP[str(card.suit).split('.')[-1]]
        for lab, suits in label_to_suits.items():
            if sch in suits:
                return lab
        return 'x'
    return tok(c_high) + tok(c_low)

# =============== 可选：treys Monte-Carlo 兜底 ===============
def _mc_equity_flop(hole2_state, flop3_state, n_players=6, n_sim=200):
    if not _TREYS_OK:
        return 0.0
    def to_t(c):
        r = RANK_CHAR_MAP[str(c.rank).split('.')[-1]]
        s = SUIT_CHAR_MAP[str(c.suit).split('.')[-1]]
        return TCard.new(r + s)
    hero = [to_t(hole2_state[0]), to_t(hole2_state[1])]
    flop = [to_t(x) for x in flop3_state]
    win = tie = 0
    for _ in range(n_sim):
        d = TDeck()
        used = set(hero + flop)
        d.cards = [c for c in d.cards if c not in used]
        opps = [d.draw(2) for _ in range(n_players - 1)]
        board = flop + d.draw(2)
        hs = _teval.evaluate(board, hero)
        os = [_teval.evaluate(board, o) for o in opps]
        best = min([hs] + os)
        winners = [s for s in [hs] + os if s == best]
        if hs == best:
            if len(winners) == 1: win += 1
            else: tie += 1
    return (win + 0.5 * tie) / n_sim

def _mc_equity_turn(hole2_state, turn4_state, n_players=6, n_sim=200):
    if not _TREYS_OK:
        return 0.0
    def to_t(c):
        r = RANK_CHAR_MAP[str(c.rank).split('.')[-1]]
        s = SUIT_CHAR_MAP[str(c.suit).split('.')[-1]]
        return TCard.new(r + s)
    hero = [to_t(hole2_state[0]), to_t(hole2_state[1])]
    turn = [to_t(x) for x in turn4_state]
    win = tie = 0
    for _ in range(n_sim):
        d = TDeck()
        used = set(hero + turn)
        d.cards = [c for c in d.cards if c not in used]
        opps = [d.draw(2) for _ in range(n_players - 1)]
        river = d.draw(1)
        board = turn + river
        hs = _teval.evaluate(board, hero)
        os = [_teval.evaluate(board, o) for o in opps]
        best = min([hs] + os)
        winners = [s for s in [hs] + os if s == best]
        if hs == best:
            if len(winners) == 1: win += 1
            else: tie += 1
    return (win + 0.5 * tie) / n_sim

def _mc_equity_river(hole2_state, board5_state, n_players=6, n_sim=500):
    if not _TREYS_OK:
        return 0.0
    def to_t(c):
        r = RANK_CHAR_MAP[str(c.rank).split('.')[-1]]
        s = SUIT_CHAR_MAP[str(c.suit).split('.')[-1]]
        return TCard.new(r + s)
    hero = [to_t(hole2_state[0]), to_t(hole2_state[1])]
    board5 = [to_t(x) for x in board5_state]
    win = tie = 0
    for _ in range(n_sim):
        d = TDeck()
        used = set(hero + board5)
        d.cards = [c for c in d.cards if c not in used]
        opps = [d.draw(2) for _ in range(n_players - 1)]
        hs = _teval.evaluate(board5, hero)
        os = [_teval.evaluate(board5, o) for o in opps]
        best = min([hs] + os)
        winners = [s for s in [hs] + os if s == best]
        if hs == best:
            if len(winners) == 1: win += 1
            else: tie += 1
    return (win + 0.5 * tie) / n_sim

# =============== 模型（保持你原来的结构） ===============
class PokerNetwork(nn.Module):
    def __init__(self, input_size=500, hidden_size=512, num_actions=3):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.action_head = nn.Linear(hidden_size, num_actions)
        self.sizing_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, opponent_features=None):
        features = self.base(x)
        action_logits = self.action_head(features)
        bet_size = 0.1 + 2.9 * self.sizing_head(features)
        return action_logits, bet_size

# =============== 主编码：带街道门控的权益槽 ===============
def encode_state(state, player_id=0):
    """
    输出固定 (500,) 的特征向量：
      - 手/公牌 one-hot、阶段、彩池、位置信息、玩家状态、最小下注、合法动作、上一动作……
      - 三个权益槽：eq_flop / eq_turn / eq_river —— 仅在当前街道非零，其它置 0
      - preflop_equity 永远在最后一维，且仅在 Preflop 写入
    """
    encoded = []
    num_players = 6  # 固定 6 人桌

    # 1) 手牌 52
    hand_enc = np.zeros(52, dtype=np.float32)
    for card in state.players_state[player_id].hand:
        idx = int(card.suit) * 13 + int(card.rank)
        hand_enc[idx] = 1.0
    encoded.append(hand_enc)

    # 2) 公共牌 52
    community_enc = np.zeros(52, dtype=np.float32)
    for card in state.public_cards:
        idx = int(card.suit) * 13 + int(card.rank)
        community_enc[idx] = 1.0
    encoded.append(community_enc)

    # 3) 阶段 one-hot 5
    stage_enc = np.zeros(5, dtype=np.float32)
    stage_enc[int(state.stage)] = 1.0
    encoded.append(stage_enc)

    # 4) 彩池（归一化）1
    initial_stake = max(1.0, state.players_state[0].stake)
    encoded.append(np.array([state.pot / initial_stake], dtype=np.float32))

    # 5) 庄位 + 当前行动者（6+6）
    btn = np.zeros(num_players, dtype=np.float32); btn[state.button % num_players] = 1.0
    cur = np.zeros(num_players, dtype=np.float32); cur[state.current_player % num_players] = 1.0
    encoded.extend([btn, cur])

    # 5.5) 英雄位置（UTG/MP/CO/BTN/SB/BB） + 当前活跃玩家数
    # --------------------------------------------------------
    hero_position_index = 0
    try:
        # hero 的座位号（假设 state.button 为庄家位置，从0开始循环）
        # 按照标准6人桌位置顺序：UTG(庄家左手第一位)=0, MP=1, CO=2, BTN=3, SB=4, BB=5
        # 实际位置 = (player_id - button - 1) % num_players
        hero_position_index = (player_id - state.button - 1) % num_players
    except Exception as e:
        if VERBOSE:
            print(f"[WARN] hero_position_index compute failed: {e}")
        hero_position_index = 0

    num_active_players = 0
    try:
        num_active_players = sum(1 for ps in state.players_state if getattr(ps, "active", False))
    except Exception as e:
        if VERBOSE:
            print(f"[WARN] num_active_players compute failed: {e}")
        num_active_players = 0

    encoded.append(np.array([hero_position_index], dtype=np.float32))
    encoded.append(np.array([num_active_players], dtype=np.float32))

    # 6) 每个玩家状态（6×4）
    for p in range(num_players):
        if p < len(state.players_state):
            ps = state.players_state[p]
            active = 1.0 if ps.active else 0.0
            bet = ps.bet_chips / initial_stake
            pot_chips = ps.pot_chips / initial_stake
            stake = ps.stake / initial_stake
        else:
            active = bet = pot_chips = stake = 0.0
        encoded.append(np.array([active, bet, pot_chips, stake], dtype=np.float32))

    # 7) 最小下注 1
    encoded.append(np.array([state.min_bet / initial_stake], dtype=np.float32))

    # 8) 合法动作 4
    legal_enc = np.zeros(4, dtype=np.float32)
    for action_enum in state.legal_actions:
        idx = min(int(action_enum), 3)
        legal_enc[idx] = 1.0
    encoded.append(legal_enc)

    # 9) 上一步动作 5（4种动作 + 金额）
    prev_enc = np.zeros(5, dtype=np.float32)
    if state.from_action is not None:
        idx = min(int(state.from_action.action.action), 3)
        prev_enc[idx] = 1.0
        prev_enc[4] = state.from_action.action.amount / initial_stake
    encoded.append(prev_enc)

    # 10) 三个权益槽：先算原始值，再按街道门控
    raw_flop = raw_turn = raw_river = 0.0
    try:
        hero = state.players_state[player_id].hand
        board = state.public_cards
        htype = _hand_type_169_from_state_cards(hero[0], hero[1]) if len(hero) == 2 else None

        # flop 原始值
        if len(board) >= 3 and htype is not None:
            canon_f, pat_f, seen_f = _canonicalize_flop_from_state(board[:3])
            sig_f = _suit_sig_vs_seen_for_hole(hero, seen_f)
            key4_f = (htype, _norm_canon_lower(canon_f), pat_f.lower(), sig_f.lower())
            v = _get_flop_lut().get(key4_f)
            if v is not None:
                raw_flop = float(v)
            elif _TREYS_OK:
                raw_flop = _mc_equity_flop(hero, board[:3], n_players=num_players, n_sim=200)

        # turn 原始值
        if len(board) >= 4 and htype is not None:
            canon_t, pat_t, seen_t = _canonicalize_turn_from_state(board[:4])
            sig_t = _suit_sig_vs_seen_for_hole(hero, seen_t)
            key4_t = (htype, _norm_canon_lower(canon_t), pat_t.lower(), sig_t.lower())
            v = _get_turn_lut().get(key4_t)
            if v is not None:
                raw_turn = float(v)
            elif _TREYS_OK:
                raw_turn = _mc_equity_turn(hero, board[:4], n_players=num_players, n_sim=200)

        # river 原始值（仅 MC）
        if len(board) == 5 and _TREYS_OK:
            raw_river = _mc_equity_river(hero, board[:5], n_players=num_players, n_sim=500)

    except Exception as e:
        if VERBOSE:
            print(f"[WARN] equity compute failed: {e}")
        raw_flop = raw_turn = raw_river = 0.0

    # 街道门控
    try:
        STAGE_PREFLOP = pkrs.Stage.Preflop
        STAGE_FLOP    = pkrs.Stage.Flop
        STAGE_TURN    = pkrs.Stage.Turn
        STAGE_RIVER   = pkrs.Stage.River
    except Exception:
        STAGE_PREFLOP, STAGE_FLOP, STAGE_TURN, STAGE_RIVER = 0, 1, 2, 3

    stage_now = state.stage
    eq_flop  = raw_flop  if stage_now == STAGE_FLOP  else 0.0
    eq_turn  = raw_turn  if stage_now == STAGE_TURN  else 0.0
    eq_river = raw_river if stage_now == STAGE_RIVER else 0.0

    if VERBOSE:
        print(f"[EQ(raw)] F={raw_flop:.3f} T={raw_turn:.3f} R={raw_river:.3f} | "
              f"[EQ(out)] F={eq_flop:.3f} T={eq_turn:.3f} R={eq_river:.3f}")

    encoded.append(np.array([eq_flop ], dtype=np.float32))
    encoded.append(np.array([eq_turn ], dtype=np.float32))
    encoded.append(np.array([eq_river], dtype=np.float32))

    # 11) hand_straighty_potential（当前街道对手已成顺的概率）
    straighty_prob = 0.0
    try:
        board_cards = state.public_cards
        hero_cards = state.players_state[player_id].hand

        # 只在有公共牌（Flop 之后）时计算，Preflop 固定为 0
        if len(board_cards) >= 3 and len(hero_cards) == 2:
            # 当前仍在局里的对手数量（active=True，且不是自己）
            n_opponents = 0
            for i, ps in enumerate(state.players_state):
                if i == player_id:
                    continue
                if getattr(ps, "active", False):
                    n_opponents += 1

            if n_opponents > 0:
                straighty_prob = hand_straighty_potential(
                    hero_cards,
                    board_cards,
                    n_opponents
                )

        if VERBOSE:
            print(f"[STRAIGHTY] hand_straighty_potential = {straighty_prob:.4f}")

    except Exception as e:
        if VERBOSE:
            print(f"[WARN] hand_straighty_potential compute failed: {e}")
        straighty_prob = 0.0

    encoded.append(np.array([straighty_prob], dtype=np.float32))

    # 统计当前仍在局里的对手人数
    flushy_prob = 0.0
    try:
        hero_cards = state.players_state[player_id].hand
        n_opponents = sum(
            1
            for i, ps in enumerate(state.players_state)
            if i != player_id and getattr(ps, "active", False)
        )

        flushy_prob = 0.0
        if len(state.public_cards) >= 3 and len(hero_cards) == 2 and n_opponents > 0:
            flushy_prob = hand_flushy_potential(hero_cards, state.public_cards, n_opponents)
    except Exception as e:
        if VERBOSE:
            print(f"[WARN] hand_flush_potential compute failed: {e}")
            flushy_prob = 0.0
    encoded.append(np.array([flushy_prob], dtype=np.float32))

    # 11) highcard_on_board_norm（公共牌最高牌归一化：A = 1.0）
    highcard_norm = 0.0
    try:
        highcard_norm = highcard_on_board_norm(state.public_cards)
    except Exception as e:
        if VERBOSE:
            print(f"[WARN] highcard_on_board_norm compute failed: {e}")
        highcard_norm = 0.0
    encoded.append(np.array([highcard_norm], dtype=np.float32))

    lowcard_norm = 0.0
    try:
        lowcard_norm = lowcard_on_board_norm(state.public_cards)
    except Exception as e:
        if VERBOSE:
            print(f"[WARN] lowcard_on_board_norm compute failed: {e}")
        lowcard_norm = 0.0
    encoded.append(np.array([lowcard_norm], dtype=np.float32))

    # 11) board_gap（公共牌最高牌归一化：A = 1.0）
    board_gap = 0.0
    try:
        board_gap = board_gap_norm(state.public_cards)
    except Exception as e:
        if VERBOSE:
            print(f"[WARN] board_gap_norm compute failed: {e}")
        board_gap = 0.0
    encoded.append(np.array([board_gap], dtype=np.float32))

    # 11) board_gap（公共牌最高牌归一化：A = 1.0）
    avg_rank = 0.0
    try:
        avg_rank = avg_rank_on_board_norm(state.public_cards)
    except Exception as e:
        if VERBOSE:
            print(f"[WARN] avg_rank compute failed: {e}")
        avg_rank = 0.0
    encoded.append(np.array([avg_rank], dtype=np.float32))

    # 11) board_gap（公共牌最高牌归一化：A = 1.0）
    paired_level_c = 0.0
    try:
        paired_level_c = paired_level(state.public_cards)
    except Exception as e:
        if VERBOSE:
            print(f"[WARN] paired_level compute failed: {e}")
        paired_level_c = 0.0
    encoded.append(np.array([paired_level_c], dtype=np.float32))

    # 11) s_hint（公共牌最高牌归一化：A = 1.0）
    s_hint = 0.0
    try:
        s_hint = straighty_hint(state.public_cards)
    except Exception as e:
        if VERBOSE:
            print(f"[WARN] straighty_hint compute failed: {e}")
        s_hint = 0.0
    encoded.append(np.array([s_hint], dtype=np.float32))

    # 11) f_hint（公共牌最高牌归一化：A = 1.0）
    f_hint = 0.0
    try:
        f_hint = flushy_hint(state.public_cards)
    except Exception as e:
        if VERBOSE:
            print(f"[WARN] flushy_hint compute failed: {e}")
        f_hint = 0.0
    encoded.append(np.array([f_hint], dtype=np.float32))

    # 11) f_board（公共牌最高牌归一化：A = 1.0）
    f_board = 0.0
    try:
        f_board = flush_on_board(state.public_cards)
    except Exception as e:
        if VERBOSE:
            print(f"[WARN] flush_on_board compute failed: {e}")
        f_board = 0.0
    encoded.append(np.array([f_board], dtype=np.float32))

    monotone_v = 0.0
    try:
        monotone_v = monotone(state.public_cards)
    except Exception as e:
        if VERBOSE:
            print(f"[WARN] monotone compute failed: {e}")
        monotone_v = 0.0
    encoded.append(np.array([monotone_v], dtype=np.float32))

    two_tone_v = 0.0
    try:
        two_tone_v = two_tone(state.public_cards)
    except Exception as e:
        if VERBOSE:
            print(f"[WARN] two_tone compute failed: {e}")
        two_tone_v = 0.0
    encoded.append(np.array([two_tone_v], dtype=np.float32))

    rainbow_v = 0.0
    try:
        rainbow_v = rainbow(state.public_cards)
    except Exception as e:
        if VERBOSE:
            print(f"[WARN] rainbow compute failed: {e}")
        rainbow_v = 0.0
    encoded.append(np.array([rainbow_v], dtype=np.float32))


    # SPR：Stack-to-Pot Ratio（0~10），表示剩余筹码压力
    spr_v = 0.0
    try:
        pot_size = state.pot
        hero_state = state.players_state[player_id]
        hero_stack = getattr(hero_state, "stake", 0.0)

        # 所有仍在局内的对手筹码（取最短作为有效筹码的对手那一端）
        opponent_stacks = [
            getattr(ps, "stake", 0.0)
            for i, ps in enumerate(state.players_state)
            if i != player_id and getattr(ps, "active", False)
        ]

        if pot_size > 0 and opponent_stacks:
            villain_stack = min(opponent_stacks)
            spr_v = spr(float(hero_stack), float(villain_stack), float(pot_size))

        if VERBOSE:
            print(f"[SPR] hero_stack={hero_stack}, "
                  f"min_opp_stack={min(opponent_stacks) if opponent_stacks else 0}, "
                  f"pot={pot_size}, spr={spr_v:.3f}")
    except Exception as e:
        if VERBOSE:
            print(f"[WARN] spr compute failed: {e}")
        spr_v = 0.0

    encoded.append(np.array([spr_v], dtype=np.float32))

    flush_suits_count = 0
    try:
        flush_suits_count = board_flush_possible_suits(state.public_cards)
    except Exception as e:
        if VERBOSE:
            print(f"[WARN] board_flush_possible_suits compute failed: {e}")
        flush_suits_count = 0
    encoded.append(np.array([flush_suits_count], dtype=np.float32))

    # ==== implied_pot_odds_hint：当前听牌投资回报比估计 ====
    implied_pot_odds_hint = 0.0
    try:
        pot_size = state.pot
        hero_state = state.players_state[player_id]
        hero_stack = getattr(hero_state, "stake", 0.0)

        # 当前 hero 需要跟注多少（max_bet - hero 已下的 bet）
        to_call = 0.0
        try:
            current_bet = max(ps.bet_chips for ps in state.players_state)
            to_call = max(0.0, current_bet - hero_state.bet_chips)
        except Exception:
            to_call = 0.0

        # 有效筹码（和 SPR 一样，用最短那一方）
        opponent_stacks = [
            getattr(ps, "stake", 0.0)
            for i, ps in enumerate(state.players_state)
            if i != player_id and getattr(ps, "active", False)
        ]
        villain_stack = min(opponent_stacks) if opponent_stacks else 0.0
        effective_stack = min(hero_stack, villain_stack)

        #    - Preflop：用 preflop_equity（get_preflop_equity）
        #    - Flop / Turn / River：用 raw_flop / raw_turn / raw_river
        if stage_now == STAGE_PREFLOP:
            # 用你上面已经定义好的查表函数
            try:
                current_equity = float(get_preflop_equity(hero_state.hand))
            except Exception:
                current_equity = 0.0
        elif stage_now == STAGE_FLOP:
            current_equity = float(raw_flop)
        elif stage_now == STAGE_TURN:
            current_equity = float(raw_turn)
        elif stage_now == STAGE_RIVER:
            current_equity = float(raw_river)
        else:
            current_equity = 0.0  # 其他异常阶段就当 0

        alpha = 0.3  # 命中之后，期望还能从有效筹码中再赢到 alpha 部分
        implied_pot = pot_size + to_call + alpha * effective_stack

        if implied_pot > 0 and to_call > 0 and current_equity > 0:
            # 所需 equity（含 implied odds）
            implied_pot_odds = to_call / implied_pot  # C / (P + C + implied)

            # 性价比比值：equity / required_equity
            ratio = current_equity / implied_pot_odds if implied_pot_odds > 0 else 0.0

            # 压缩到 0~1：ratio=1 → 0.5，ratio>=2 → ~1
            implied_pot_odds_hint = max(0.0, min(1.0, ratio / 2.0))
        else:
            implied_pot_odds_hint = 0.0

        if VERBOSE:
            print(f"[IMPLIED_POT_ODDS] stage={stage_now}, eq={current_equity:.3f}, "
                  f"to_call={to_call:.2f}, pot={pot_size:.2f}, eff_stack={effective_stack:.2f}, "
                  f"hint={implied_pot_odds_hint:.3f}")
    except Exception as e:
        if VERBOSE:
            print(f"[WARN] implied_pot_odds_hint compute failed: {e}")
        implied_pot_odds_hint = 0.0

    encoded.append(np.array([implied_pot_odds_hint], dtype=np.float32))

    # 11) preflop_equity（最后一维；仅 Preflop 写入）
    preflop_equity = 0.0
    if len(state.public_cards) == 0:  # Preflop
        ranks = [str(c.rank).split('.')[-1] for c in state.players_state[player_id].hand]
        suits = [str(c.suit).split('.')[-1] for c in state.players_state[player_id].hand]
        rank_map = {'R2':'2','R3':'3','R4':'4','R5':'5','R6':'6','R7':'7','R8':'8',
                    'R9':'9','RT':'T','RJ':'J','RQ':'Q','RK':'K','RA':'A'}
        ranks_char = [rank_map[r] for r in ranks]
        suited = 's' if suits[0] == suits[1] else 'o'
        hand_str = f"{ranks_char[0]}{ranks_char[1]}{suited}".upper()
        row = equity_table.loc[equity_table['hand'] == hand_str]
        if not row.empty:
            preflop_equity = float(row['equity'].values[0])
        if VERBOSE:
            print(f"[DEBUG] preflop slot equity {hand_str} = {preflop_equity:.6f}")

    encoded.append(np.array([preflop_equity], dtype=np.float32))

    # 12) 拼接为 (500,)
    x = np.concatenate(encoded).astype(np.float32, copy=False)
    if x.shape[0] < 500:
        x = np.pad(x, (0, 500 - x.shape[0]), mode='constant')
    elif x.shape[0] > 500:
        raise ValueError(f"[ERROR] Encoded vector too long ({x.shape[0]}).")
    return x
