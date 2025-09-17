"""
カードゲーム「101」のAIエージェントを強化学習で訓練するスクリプト。

Transformer ベースのモデルを用いて自己対戦から最適戦略を学習します。

学習データは以下のディレクトリに保存されます:
- エピソードデータ: <プロジェクトルート>/data/episodes/
- モデルの重み: <プロジェクトルート>/models/
"""

# mypy: disallow-subclassing-any=False

import logging
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from one_o_one.game import Action, Card, State, action_mask, legal_actions, reset, step

logger = logging.getLogger(__name__)

# 将来の非推奨に関する警告を非表示にする
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

# --- 定数と設定 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EPISODE_DATA_DIR = PROJECT_ROOT / "data" / "episodes"
MODEL_SAVE_DIR = PROJECT_ROOT / "models"

# 訓練パラメータ
NUM_EPISODES = 1  # 訓練エピソード数
REPLAY_BUFFER_SIZE = 10000  # リプレイバッファのサイズ
EPS_START = 1.0  # ε-greedy法の開始ε
EPS_END = 0.01
EPS_DECAY = 0.995
LEARNING_RATE = 0.001
NUM_PLAYERS = 4
HISTORY_LENGTH = 48
ACTION_SIZE = len(Action)  # アクションの数
# Opponent hand prediction adds 32 values per opponent (2 cards * 16 ranks)
STATE_SIZE = 35 + (NUM_PLAYERS + ACTION_SIZE) * HISTORY_LENGTH + (NUM_PLAYERS - 1) * 32

# 入力に使用する公開情報ベクトルのサイズ (total, direction, penalty, history)
PUBLIC_STATE_SIZE = 3 + (NUM_PLAYERS + ACTION_SIZE) * HISTORY_LENGTH


# --- 状態のベクトル化 ---


def _to_one_hot(num: int, max_val: int) -> list[float]:
    """指定された値をOne-Hotエンコーディングする。"""
    return [1.0 if i == num else 0.0 for i in range(max_val)]


def _rank_value(card: Card | None) -> int:
    return int(card.rank) if card is not None else 0


def encode_state(s: State) -> np.ndarray:
    """単一のゲーム状態をベクトルに変換する。"""
    me = s.players[s.public.turn]
    hand_vec = _to_one_hot(_rank_value(me.hand[0]), 16) + _to_one_hot(
        _rank_value(me.hand[1]), 16
    )
    total_vec = [s.public.total / 101.0]
    dir_vec = [1.0 if s.public.direction == 1 else 0.0]
    penalty_vec = [(s.public.penalty_level - 1) / 5.0]
    history_vec: list[float] = []
    recent = s.public.history[-HISTORY_LENGTH:]
    for player_idx, act in recent:
        history_vec.extend(_to_one_hot(player_idx, NUM_PLAYERS))
        history_vec.extend(_to_one_hot(act, ACTION_SIZE))
    missing = HISTORY_LENGTH - len(recent)
    history_vec.extend([0.0] * (missing * (NUM_PLAYERS + ACTION_SIZE)))

    state_vector = np.array(
        hand_vec + total_vec + dir_vec + penalty_vec + history_vec,
        dtype=np.float32,
    )
    return state_vector


def encode_public_state(s: State) -> np.ndarray:
    """公開情報のみをベクトル化したものを返す。"""
    total_vec = [s.public.total / 101.0]
    dir_vec = [1.0 if s.public.direction == 1 else 0.0]
    penalty_vec = [(s.public.penalty_level - 1) / 5.0]
    history_vec: list[float] = []
    recent = s.public.history[-HISTORY_LENGTH:]
    for player_idx, act in recent:
        history_vec.extend(_to_one_hot(player_idx, NUM_PLAYERS))
        history_vec.extend(_to_one_hot(act, ACTION_SIZE))
    missing = HISTORY_LENGTH - len(recent)
    history_vec.extend([0.0] * (missing * (NUM_PLAYERS + ACTION_SIZE)))
    return np.array(total_vec + dir_vec + penalty_vec + history_vec, dtype=np.float32)


def get_vector(
    states: list[State],
    actions: list[int],
    rewards: list[float],
    agent: "TransformerAgent | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ゲーム開始からの状態・行動・報酬のシーケンスを返す。"""
    state_vecs: list[np.ndarray] = []
    for st in states:
        vec = encode_state(st)
        if agent is not None:
            vec = np.concatenate([vec, agent.predict_opponents(st)])
        state_vecs.append(vec)
    action_seq = actions + [0]
    reward_seq = rewards + [0.0]
    return (
        np.array(state_vecs, dtype=np.float32),
        np.array(action_seq, dtype=np.int64),
        np.array(reward_seq, dtype=np.float32),
    )


# --- Transformerベースのモデル ---


class DecisionTransformer(nn.Module):
    """状態・行動・報酬の系列から次の行動を予測するTransformerモデル。"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.state_embed = nn.Linear(state_size, hidden_size)
        self.action_embed = nn.Embedding(action_size, hidden_size)
        self.reward_embed = nn.Linear(1, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.predict_head = nn.Linear(hidden_size, action_size)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """各時刻の状態・行動・報酬から次の行動のロジットを計算する。"""
        state_tokens = self.state_embed(states)
        action_tokens = self.action_embed(actions)
        reward_tokens = self.reward_embed(rewards.unsqueeze(-1))
        tokens = state_tokens + action_tokens + reward_tokens
        tokens = tokens.unsqueeze(1)  # (T, 1, hidden)
        out = self.encoder(tokens)
        logits: torch.Tensor = self.predict_head(out.squeeze(1))
        return logits


# --- 相手モデリングネットワーク ---


class OpponentModel(nn.Module):
    """公開情報から各プレイヤーの手札を推定するモデル。"""

    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2 * 16)  # 2枚のカードそれぞれ16通り

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        logits: torch.Tensor = self.fc2(h)
        return logits.view(-1, 2, 16)


# --- AIエージェント ---


class TransformerAgent:
    """DecisionTransformer を用いたエージェント。"""

    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        self.model = DecisionTransformer(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.opponent_model = OpponentModel(PUBLIC_STATE_SIZE + NUM_PLAYERS).to(
            self.device
        )
        self.opponent_optimizer = optim.Adam(
            self.opponent_model.parameters(), lr=LEARNING_RATE
        )
        self.epsilon = EPS_START
        self.memory: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.opponent_memory: list[tuple[np.ndarray, np.ndarray]] = []

    def select_action(
        self,
        state: State,
        state_seq: np.ndarray,
        action_seq: np.ndarray,
        reward_seq: np.ndarray,
    ) -> Action:
        """ε-greedy法に基づいて合法手から行動を選択する。"""
        if random.random() < self.epsilon:
            legal = legal_actions(state)
            return random.choice(legal)

        with torch.no_grad():
            states_t = torch.FloatTensor(state_seq).to(self.device)
            actions_t = torch.LongTensor(action_seq).to(self.device)
            rewards_t = torch.FloatTensor(reward_seq).to(self.device)
            logits: torch.Tensor = self.model(states_t, actions_t, rewards_t)[-1]
            mask = torch.tensor(
                action_mask(state), dtype=torch.bool, device=self.device
            )
            logits[~mask] = -1e9
            return Action(int(logits.argmax().item()))

    def predict_opponents(self, state: State) -> np.ndarray:
        """現在の状態から他プレイヤーの手札分布を推定する。"""
        public_vec = encode_public_state(state)
        preds: list[np.ndarray] = []
        with torch.no_grad():
            pub_t = torch.FloatTensor(public_vec).to(self.device)
            for idx in range(NUM_PLAYERS):
                if idx == state.public.turn:
                    continue
                inp = torch.cat(
                    (
                        pub_t,
                        torch.FloatTensor(_to_one_hot(idx, NUM_PLAYERS)).to(
                            self.device
                        ),
                    )
                )
                logits = self.opponent_model(inp.unsqueeze(0)).squeeze(0)
                probs = torch.softmax(logits, dim=-1)
                preds.append(probs.view(-1).cpu().numpy())
        return np.concatenate(preds)

    def store_episode(
        self,
        state_seq: np.ndarray,
        action_seq: np.ndarray,
        reward_seq: np.ndarray,
        final_state: State,
    ) -> None:
        self.memory.append((state_seq, action_seq, reward_seq))
        if len(self.memory) > REPLAY_BUFFER_SIZE:
            self.memory.pop(0)
        public_vec = encode_public_state(final_state)
        for idx, player in enumerate(final_state.players):
            feature = np.concatenate(
                [public_vec, _to_one_hot(idx, NUM_PLAYERS)]
            ).astype(np.float32)
            label = np.array(
                [
                    _rank_value(player.hand[0]),
                    _rank_value(player.hand[1]),
                ],
                dtype=np.int64,
            )
            self.opponent_memory.append((feature, label))
        if len(self.opponent_memory) > REPLAY_BUFFER_SIZE:
            self.opponent_memory.pop(0)

    def learn(self) -> None:
        """軌跡データを用いてモデルを更新する。"""
        if not self.memory:
            return
        states, actions, rewards = random.choice(self.memory)
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        logits = self.model(states_t, actions_t, rewards_t)[:-1]
        loss: torch.Tensor = nn.functional.cross_entropy(logits, actions_t[1:])
        self.optimizer.zero_grad()
        # PyTorch の Tensor.backward は型ヒントが提供されていないため
        # 明示的に torch.autograd.backward を使用して勾配を計算する。
        torch.autograd.backward(loss)
        self.optimizer.step()

        if self.opponent_memory:
            features, labels = random.choice(self.opponent_memory)
            feat_t = torch.FloatTensor(features).to(self.device)
            label_t = torch.LongTensor(labels).to(self.device)
            logits = self.opponent_model(feat_t.unsqueeze(0)).squeeze(0)
            opp_loss: torch.Tensor = nn.functional.cross_entropy(logits, label_t)
            self.opponent_optimizer.zero_grad()
            torch.autograd.backward(opp_loss)
            self.opponent_optimizer.step()

    def update_epsilon(self) -> None:
        """εを減衰させる。"""
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)


# --- 自己対戦と学習ループ ---


def main() -> None:
    """学習のメインループを実行する。"""
    # 保存用ディレクトリを作成
    EPISODE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    agent = TransformerAgent(STATE_SIZE, ACTION_SIZE)
    model_path = MODEL_SAVE_DIR / "101_transformer.pth"
    if model_path.exists():
        logger.info("Loading model from %s", model_path)
        agent.model.load_state_dict(torch.load(model_path))

    all_episode_data = []

    for episode in range(NUM_EPISODES):
        s: State = reset(4)
        episode_reward = 0.0
        episode_data: list[dict[str, object]] = []
        done = False
        state_history: list[State] = []
        action_history: list[int] = []
        reward_history: list[float] = []

        while not done:
            state_history.append(s)
            state_seq, action_seq, reward_seq = get_vector(
                state_history, action_history, reward_history, agent
            )
            action = agent.select_action(s, state_seq, action_seq, reward_seq)

            next_s, reward, done, _ = step(s, action)
            action_history.append(action.value)
            reward_history.append(reward)

            episode_reward += reward
            episode_data.append({"state": encode_state(s), "action": action.value})
            s = next_s

        state_history.append(s)
        state_seq, action_seq, reward_seq = get_vector(
            state_history, action_history, reward_history, agent
        )
        agent.store_episode(state_seq, action_seq, reward_seq, s)
        agent.learn()
        all_episode_data.extend(episode_data)
        agent.update_epsilon()

        if episode % 100 == 0:
            logger.info(
                "Episode %d, Total Reward: %s, Epsilon: %.4f",
                episode,
                episode_reward,
                agent.epsilon,
            )
            # モデルを保存
            torch.save(agent.model.state_dict(), model_path)

            # エピソードデータをParquet形式で保存
            df = pd.DataFrame(all_episode_data)
            df.to_parquet(EPISODE_DATA_DIR / f"episode_data_{episode}.parquet")
            all_episode_data = []

    logger.info("Training finished.")
    torch.save(agent.model.state_dict(), model_path)
    logger.info("Final model saved to %s", model_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
