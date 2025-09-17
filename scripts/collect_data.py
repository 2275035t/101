"""
学習済みAIエージェント同士を対戦させ、その棋譜（ゲームログ）を収集するスクリプト。

このスクリプトは、ローカルに保存されたTransformerベースの学習済みモデルを
4体のエージェントにロードし、シミュレーションを実行します。
生成されたデータは、観戦用アニメーション（spectate.py）の入力として使用できます。

- モデル読込元: <プロジェクトルート>/models/101_transformer.pth
- データ保存先: <プロジェクトルート>/data/game_data_for_spectate.json
"""

# mypy: disallow-subclassing-any=False

import json
import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch

from train_ai import (
    ACTION_SIZE,
    NUM_PLAYERS,
    PUBLIC_STATE_SIZE,
    STATE_SIZE,
    DecisionTransformer,
    OpponentModel,
    to_one_hot,
    encode_public_state,
    get_vector,
)

from one_o_one.game import Action, Rank, State, action_mask, reset, step

logger = logging.getLogger(__name__)

# --- 定数と設定 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "101_transformer.pth"
OUTPUT_DATA_PATH = PROJECT_ROOT / "data" / "game_data_for_spectate.json"

NUM_GAMES = 1  # 生成するゲーム数


# --- AIモデルと状態ベクトル化関数 (train_ai.pyから再利用) ---


class LogEntry(TypedDict):
    """観戦用ログの1ターン分のデータ構造。"""

    turn: int
    player: int
    total_before: int
    action: str
    played_card: str
    total_after: int
    reward: float
    done: bool
    direction: int
    penalty: int
    lp_before_json: list[int]
    hands_before_json: list[list[str]]


class AIAgent:
    """学習済みTransformerモデルを使用して行動を選択するエージェント。"""

    def __init__(self, model_path: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DecisionTransformer(STATE_SIZE, ACTION_SIZE).to(self.device)
        self.opponent_model = OpponentModel(PUBLIC_STATE_SIZE + NUM_PLAYERS).to(
            self.device
        )

        if not model_path.exists():
            raise FileNotFoundError(f"学習済みモデルが見つかりません: {model_path}")

        logger.info("Loading model from: %s", model_path)

        checkpoint = torch.load(model_path, map_location=self.device)

        if (
            "model_state_dict" not in checkpoint
            or "opponent_model_state_dict" not in checkpoint
        ):
            raise ValueError(
                "Cannot collect data. The model file is in an old format. Please retrain the agent with the updated train_ai.py script."
            )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.opponent_model.load_state_dict(checkpoint["opponent_model_state_dict"])

        self.model.eval()
        self.opponent_model.eval()

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
                        torch.FloatTensor(to_one_hot(idx, NUM_PLAYERS)).to(self.device),
                    )
                )
                logits = self.opponent_model(inp.unsqueeze(0)).squeeze(0)
                probs = torch.softmax(logits, dim=-1)
                preds.append(probs.view(-1).cpu().numpy())
        return np.concatenate(preds)

    def select_action(
        self,
        state: State,
        state_seq: np.ndarray,
        action_seq: np.ndarray,
        reward_seq: np.ndarray,
    ) -> Action:
        """現在の状態で最も評価値が高い行動を選択する。"""
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


# --- データ収集メイン処理 ---


def collect_game_data(agent: AIAgent, num_games: int) -> list[LogEntry]:
    """指定された数のゲームシミュレーションを行い、ログを収集する。"""
    all_games_data: list[LogEntry] = []
    for i in range(num_games):
        logger.info("Simulating game %d/%d...", i + 1, num_games)
        s = reset(NUM_PLAYERS)
        game_log: list[LogEntry] = []
        done = False
        turn = 0
        state_history: list[State] = []
        action_history: list[int] = []
        reward_history: list[float] = []

        while not done:
            state_history.append(s)
            state_seq, action_seq, reward_seq = get_vector(
                state_history, action_history, reward_history, agent=agent
            )
            action = agent.select_action(s, state_seq, action_seq, reward_seq)
            next_s, reward, done, _ = step(s, action)
            action_history.append(action.value)
            reward_history.append(reward)

            cur_idx = s.public.turn
            played = "NONE"

            if action in (Action.PLAY_HAND_0, Action.PLAY_HAND_1):
                card = s.players[cur_idx].hand[action.value]
                if card is not None:
                    played = card.rank.name
            elif action == Action.PLAY_DECK:
                played = "DECK"
            elif action in (Action.PLAY_TEN_PLUS, Action.PLAY_TEN_MINUS):
                played = Rank.R10.name
            elif action in (Action.PLAY_ACE_ONE, Action.PLAY_ACE_ELEVEN):
                played = Rank.A.name

            log_entry: LogEntry = {
                "turn": turn,
                "player": cur_idx,
                "total_before": s.public.total,
                "action": action.name,
                "played_card": played,
                "total_after": next_s.public.total,
                "reward": reward,
                "done": done,
                "direction": next_s.public.direction,
                "penalty": next_s.public.penalty_level,
                "lp_before_json": [p.lp for p in s.players],
                "hands_before_json": [
                    [c.rank.name if c is not None else "None" for c in p.hand]
                    for p in s.players
                ],
            }
            game_log.append(log_entry)
            s = next_s
            turn += 1

        all_games_data.extend(game_log)
        logger.info("Game %d finished after %d turns.", i + 1, turn)

    return all_games_data


def main() -> None:
    """メイン関数：モデルをロードし、データ収集を実行して保存する。"""
    OUTPUT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        agent = AIAgent(MODEL_PATH)
        game_data = collect_game_data(agent, NUM_GAMES)

        with open(OUTPUT_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(game_data, f, indent=4)

        logger.info("Successfully collected and saved data for %d game(s).", NUM_GAMES)
        logger.info("Data saved to: %s", OUTPUT_DATA_PATH)

    except (FileNotFoundError, ValueError) as e:
        logger.error("Error: %s", e)
        logger.error(
            "Please run the training script (train_ai.py) first to generate the "
            "model file in the correct format."
        )
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
