import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pokers as pkrs

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
EQUITY_PATH = '/home/harry/deepcfr/equity_table.csv'
# 你的 flop / turn 胜率表（和 test_lookup 用的同一份）
FLOP_CSV_PATH = '/home/harry/deepcfr/flop_strength_table_nsim500.csv'
TURN_CSV_PATH = '/home/harry/deepcfr/turn_strength_table_nsim500.csv'

# 加载 Equity 表（一次性全局加载）
equity_table = pd.read_csv(EQUITY_PATH, dtype={'hand': str, 'equity': float})
equity_table['hand'] = equity_table['hand'].astype(str).str.strip().str.upper()

# rank/suit 映射常量（用于 52 维 one-hot）
RANK_MAP = {'R2': 1, 'R3': 2, 'R4': 3, 'R5': 4, 'R6': 5, 'R7': 6,
            'R8': 7, 'R9': 8, 'RT': 9, 'RJ': 10, 'RQ': 11, 'RK': 12, 'RA': 13}
SUIT_MAP = {'Spades': 0, 'Hearts': 1, 'Diamonds': 2, 'Clubs': 3}

# 供 hand_type 与 canonical 使用的字符映射
RANK_CHAR_MAP = {'R2': '2', 'R3': '3', 'R4': '4', 'R5': '5', 'R6': '6', 'R7': '7',
                 'R8': '8', 'R9': '9', 'RT': 'T', 'RJ': 'J', 'RQ': 'Q', 'RK': 'K', 'RA': 'A'}
SUIT_CHAR_MAP = {'Spades': 's', 'Hearts': 'h', 'Diamonds': 'd', 'Clubs': 'c'}

_RANKS_ORDER = "23456789TJQKA"  # 用于比较与 canonical（2..14）

def set_verbose(verbose_mode: bool):
    global VERBOSE
    VERBOSE = verbose_mode


def encode_cards(cards):
    """将牌面列表编码为 52 维 one-hot 向量（保持你原有的索引方案）"""
    vec = np.zeros(52)
    for card in cards:
        rank_str = str(card.rank).split('.')[-1]
        suit_str = str(card.suit).split('.')[-1]
        card_idx = SUIT_MAP[suit_str] * 13 + RANK_MAP[rank_str]
        vec[card_idx] = 1
    return vec


def get_preflop_equity(hand_cards):
    """根据两张手牌查表获得 preflop equity"""
    if not hand_cards:
        return 0.0

    ranks = [str(c.rank).split('.')[-1] for c in hand_cards]
    suits = [str(c.suit).split('.')[-1] for c in hand_cards]

    rank_order = list(RANK_MAP.keys())
    sorted_cards = sorted(zip(ranks, suits), key=lambda x: rank_order.index(x[0]), reverse=True)
    rank1, suit1 = sorted_cards[0]
    rank2, suit2 = sorted_cards[1]

    r1 = RANK_CHAR_MAP[rank1]
    r2 = RANK_CHAR_MAP[rank2]
    suited = 's' if suit1 == suit2 else 'o'
    hand_str = f"{r1}{r2}{suited}".upper()

    match = equity_table[equity_table['hand'] == hand_str]
    if not match.empty:
        equity = float(match['equity'].values[0])
        if VERBOSE:
            print(f"[DEBUG] Found equity for {hand_str}: {equity:.6f}")
        return equity

    if VERBOSE:
        print(f"[WARN] Hand {hand_str} not found in equity table.")
    return 0.0


# ===================== LUT（flop/turn）加载 & 规范化工具 =====================
def _norm_canon_lower(canon_text: str) -> str:
    """把 'ranks|suits' 统一小写 suits 段，去空白"""
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
            print(f"[WARN] cannot read {street} csv {path}: {e}")
        return {}

    if street == 'flop':
        df["hand_type"]      = df["hand_type"].astype(str).str.strip().str.upper()
        df["flop_pattern"]   = df["flop_pattern"].astype(str).str.strip().str.lower()
        df["suit_signature"] = df["suit_signature"].astype(str).str.strip().str.lower()
        df["canon_flop"]     = df["canon_flop"].apply(_norm_canon_lower)
        key_cols = ("hand_type", "canon_flop", "flop_pattern", "suit_signature")
    else:
        df["hand_type"]      = df["hand_type"].astype(str).str.strip().str.upper()
        df["turn_pattern"]   = df["turn_pattern"].astype(str).str.strip().str.lower()
        df["suit_signature"] = df["suit_signature"].astype(str).str.strip().str.lower()
        df["canon_turn"]     = df["canon_turn"].apply(_norm_canon_lower)
        key_cols = ("hand_type", "canon_turn", "turn_pattern", "suit_signature")

    lut = {}
    for _, r in df.iterrows():
        key = tuple(r[k] for k in key_cols)
        lut[key] = float(r["strength_mean"])
    return lut

# 懒加载（只读一次）
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

# ==== 从状态卡片生成 hand_type（169 类） ====
def _hand_type_169_from_state_cards(c1, c2) -> str:
    r1 = RANK_CHAR_MAP[str(c1.rank).split('.')[-1]]
    r2 = RANK_CHAR_MAP[str(c2.rank).split('.')[-1]]
    s1 = SUIT_CHAR_MAP[str(c1.suit).split('.')[-1]]
    s2 = SUIT_CHAR_MAP[str(c2.suit).split('.')[-1]]
    # 高牌在前
    if _RANKS_ORDER.index(r1) < _RANKS_ORDER.index(r2):
        r1, r2, s1, s2 = r2, r1, s2, s1
    if r1 == r2:
        return f"{r1}{r2}"
    return f"{r1}{r2}{'s' if s1 == s2 else 'o'}".upper()

# ==== canonical & suit signature（与 test_lookup 对齐） ====
def _canonicalize_flop_from_state(board3):
    triples = []
    for c in board3:
        rch = RANK_CHAR_MAP[str(c.rank).split('.')[-1]]
        sch = SUIT_CHAR_MAP[str(c.suit).split('.')[-1]]
        rint = _RANKS_ORDER.index(rch) + 2  # 2..14
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
    return canon, pat, suit_seen  # suit_seen: {'c':0,'d':1...}

def _suit_sig_vs_seen_for_hole(hole2, suit_seen):
    # 高牌在前
    def rch(card): return RANK_CHAR_MAP[str(card.rank).split('.')[-1]]
    c_high, c_low = sorted(hole2, key=lambda c: _RANKS_ORDER.index(rch(c)), reverse=True)
    inv = {idx: set() for idx in suit_seen.values()}
    for sch, idx in suit_seen.items():
        inv[idx].add(sch)
    label_to_suits = {chr(ord('a')+k): v for k,v in inv.items()}
    def tok(card):
        sch = SUIT_CHAR_MAP[str(card.suit).split('.')[-1]]
        for lab, suits in label_to_suits.items():
            if sch in suits:
                return lab
        return 'x'
    return tok(c_high) + tok(c_low)

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

# ==== 可选：treys Monte-Carlo 兜底 ====
def _mc_equity_flop(hole2_state, flop3_state, n_players=6, n_sim=200):
    if not _TREYS_OK:
        return 0.0
    # 转 treys ints
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
        opps = [d.draw(2) for _ in range(n_players-1)]
        board = flop + d.draw(2)
        hs = _teval.evaluate(board, hero)
        os = [_teval.evaluate(board, o) for o in opps]
        best = min([hs] + os)
        winners = [s for s in [hs] + os if s == best]
        if hs == best:
            if len(winners) == 1: win += 1
            else: tie += 1
    return (win + 0.5*tie) / n_sim

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
        opps = [d.draw(2) for _ in range(n_players-1)]
        river = d.draw(1)
        board = turn + river
        hs = _teval.evaluate(board, hero)
        os = [_teval.evaluate(board, o) for o in opps]
        best = min([hs] + os)
        winners = [s for s in [hs] + os if s == best]
        if hs == best:
            if len(winners) == 1: win += 1
            else: tie += 1
    return (win + 0.5*tie) / n_sim

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
        opps = [d.draw(2) for _ in range(n_players-1)]
        hs = _teval.evaluate(board5, hero)
        os = [_teval.evaluate(board5, o) for o in opps]
        best = min([hs] + os)
        winners = [s for s in [hs] + os if s == best]
        if hs == best:
            if len(winners) == 1: win += 1
            else: tie += 1
    return (win + 0.5*tie) / n_sim


# ===================== 模型定义 =====================
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


# ===================== 主编码函数 =====================
def encode_state(state, player_id=0):
    """
    Encode a full Pokers state into a fixed-length (500,) vector.
    - 仍保持你的布局与归一化方式
    - 新增 3 个字段：eq_flop, eq_turn, eq_river（各 1 维）
    - preflop_equity 仍然是最后一维
    """
    encoded = []
    num_players = 6  # ✅ 固定6人桌，确保长度恒定

    # ---------- 1. 手牌编码 (52维 one-hot) ----------
    hand_enc = np.zeros(52)
    for card in state.players_state[player_id].hand:
        card_idx = int(card.suit) * 13 + int(card.rank)
        hand_enc[card_idx] = 1
    encoded.append(hand_enc)

    # ---------- 2. 公共牌编码 (52维 one-hot) ----------
    community_enc = np.zeros(52)
    for card in state.public_cards:
        card_idx = int(card.suit) * 13 + int(card.rank)
        community_enc[card_idx] = 1
    encoded.append(community_enc)

    # ---------- 3. 阶段编码 (5维 one-hot) ----------
    stage_enc = np.zeros(5)
    stage_enc[int(state.stage)] = 1
    encoded.append(stage_enc)

    # ---------- 4. 彩池大小 (1维，归一化) ----------
    initial_stake = max(1.0, state.players_state[0].stake)
    pot_enc = np.array([state.pot / initial_stake])
    encoded.append(pot_enc)

    # ---------- 5. 庄位与当前行动者 (6+6维 one-hot) ----------
    button_enc = np.zeros(num_players)
    button_enc[state.button % num_players] = 1
    current_enc = np.zeros(num_players)
    current_enc[state.current_player % num_players] = 1
    encoded.extend([button_enc, current_enc])

    # ---------- 6. 每个玩家状态 (6×4维) ----------
    for p in range(num_players):
        if p < len(state.players_state):
            ps = state.players_state[p]
            active = 1.0 if ps.active else 0.0
            bet = ps.bet_chips / initial_stake
            pot_chips = ps.pot_chips / initial_stake
            stake = ps.stake / initial_stake
        else:
            active = bet = pot_chips = stake = 0.0
        encoded.append(np.array([active, bet, pot_chips, stake]))

    # ---------- 7. 最小下注 (1维) ----------
    encoded.append(np.array([state.min_bet / initial_stake]))

    # ---------- 8. 合法动作 (4维固定 one-hot) ----------
    legal_enc = np.zeros(4)
    for action_enum in state.legal_actions:
        idx = min(int(action_enum), 3)
        legal_enc[idx] = 1
    encoded.append(legal_enc)

    # ---------- 9. 上一步动作 (5维固定: 4种动作 + 金额) ----------
    prev_enc = np.zeros(5)
    if state.from_action is not None:
        idx = min(int(state.from_action.action.action), 3)
        prev_enc[idx] = 1
        prev_enc[4] = state.from_action.action.amount / initial_stake
    encoded.append(prev_enc)

    # ---------- 新增：eq_flop / eq_turn / eq_river （各 1 维） ----------
    eq_flop = eq_turn = eq_river = 0.0
    try:
        hero = state.players_state[player_id].hand
        board = state.public_cards

        # 只在相应街道后才尝试计算
        if len(board) >= 3:
            # hand_type
            htype = _hand_type_169_from_state_cards(hero[0], hero[1])
            # flop canonical + signature
            canon_f, pat_f, seen_f = _canonicalize_flop_from_state(board[:3])
            sig_f = _suit_sig_vs_seen_for_hole(hero, seen_f)
            key4_f = (htype, _norm_canon_lower(canon_f), pat_f.lower(), sig_f.lower())
            val = _get_flop_lut().get(key4_f)
            if val is not None:
                eq_flop = float(val)
            elif _TREYS_OK:
                eq_flop = _mc_equity_flop(hero, board[:3], n_players=num_players, n_sim=200)

        if len(board) >= 4:
            canon_t, pat_t, seen_t = _canonicalize_turn_from_state(board[:4])
            sig_t = _suit_sig_vs_seen_for_hole(hero, seen_t)
            key4_t = (htype, _norm_canon_lower(canon_t), pat_t.lower(), sig_t.lower())
            val = _get_turn_lut().get(key4_t)
            if val is not None:
                eq_turn = float(val)
            elif _TREYS_OK:
                eq_turn = _mc_equity_turn(hero, board[:4], n_players=num_players, n_sim=200)

        if len(board) == 5 and _TREYS_OK:
            eq_river = _mc_equity_river(hero, board[:5], n_players=num_players, n_sim=500)

        if VERBOSE:
            print(f"[EQ] flop={eq_flop:.3f} turn={eq_turn:.3f} river={eq_river:.3f}")

    except Exception as e:
        if VERBOSE:
            print(f"[WARN] equity compute failed: {e}")
        eq_flop = eq_turn = eq_river = 0.0

    encoded.append(np.array([eq_flop]))
    encoded.append(np.array([eq_turn]))
    encoded.append(np.array([eq_river]))

    # ---------- 10. preflop_equity (1维固定，仍旧最后一维) ----------
    preflop_equity = 0.0
    if not state.public_cards:
        rank_map = {'R2':'2','R3':'3','R4':'4','R5':'5','R6':'6','R7':'7','R8':'8',
                    'R9':'9','RT':'T','RJ':'J','RQ':'Q','RK':'K','RA':'A'}
        ranks = [str(c.rank).split('.')[-1] for c in state.players_state[player_id].hand]
        suits = [str(c.suit).split('.')[-1] for c in state.players_state[player_id].hand]
        ranks_char = [rank_map[r] for r in ranks]
        suited = 's' if suits[0] == suits[1] else 'o'
        hand_str = f"{ranks_char[0]}{ranks_char[1]}{suited}".upper()
        match = equity_table[equity_table['hand'] == hand_str]
        if not match.empty:
            preflop_equity = float(match['equity'].values[0])
        if VERBOSE:
            print(f"[DEBUG] Found equity for {hand_str}: {preflop_equity}")

    encoded.append(np.array([preflop_equity]))

    # ---------- 11. 拼接并固定维度 ----------
    x = np.concatenate(encoded)
    base_len = len(x)

    if base_len < 500:
        x = np.pad(x, (0, 500 - base_len), mode='constant')
    elif base_len > 500:
        raise ValueError(f"[ERROR] Encoded vector too long ({base_len}) — investigate source.")

    if VERBOSE:
        print(f"[DEBUG] Final vector length: {len(x)} | (pre, f, t, r)=({preflop_equity:.3f}, {eq_flop:.3f}, {eq_turn:.3f}, {eq_river:.3f})")

    return x
