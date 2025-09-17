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
    STATE_SIZE,
    DecisionTransformer,
    get_vector,
)

from one_o_one.game import Action, State, action_mask, reset, step

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

        if not model_path.exists():
            raise FileNotFoundError(f"学習済みモデルが見つかりません: {model_path}")

        logger.info("Loading model from: %s", model_path)

        loaded = torch.load(model_path, map_location=self.device)

        # Support a variety of saved formats.  Some historical training scripts
        # stored the entire agent object (with attributes like ``policy_net``),
        # while newer versions save just the model's ``state_dict``.  To keep
        # this data-collection utility robust, attempt to extract a state
        # dictionary from whatever object was loaded.
        if isinstance(loaded, dict):
            # Common case: the raw ``state_dict`` or wrapped in a dictionary.
            if "model_state_dict" in loaded:
                state_dict = loaded["model_state_dict"]
            elif "state_dict" in loaded:
                state_dict = loaded["state_dict"]
            elif "policy_net" in loaded and isinstance(loaded["policy_net"], dict):
                state_dict = loaded["policy_net"]
            else:
                state_dict = loaded
        else:
            # Try typical attributes in priority order, guarding against
            # partially-saved agent objects whose ``state_dict`` may raise if
            # submodules like ``policy_net`` are missing.
            state_dict = None
            for obj in (
                getattr(loaded, "model", None),
                getattr(loaded, "policy_net", None),
                loaded,
            ):
                if obj is None:
                    continue
                try:
                    state_dict = obj.state_dict()
                    break
                except Exception:
                    continue
            if state_dict is None:
                raise AttributeError("Loaded object does not contain a state_dict")

        self.model.load_state_dict(state_dict)
        self.model.eval()

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
                state_history, action_history, reward_history
            )
            action = agent.select_action(s, state_seq, action_seq, reward_seq)
            next_s, reward, done, _ = step(s, action)
            action_history.append(action.value)
            reward_history.append(reward)

            cur_idx = s.public.turn
            if action in (Action.PLAY_HAND_0, Action.PLAY_HAND_1):
                card = s.players[cur_idx].hand[action.value]
                played = card.rank.name if card is not None else "NONE"
            else:
                played = "DECK"
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

    except FileNotFoundError as e:
        logger.error("Error: %s", e)
        logger.error(
            "Please run the training script (train_ai.py) first to generate the "
            "model file."
        )
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
