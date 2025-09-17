"""Deep CFR training script for the 101 card game.

This module implements a DeepCFR agent that learns a strategy close to a
Nash equilibrium via counterfactual regret minimisation.  The implementation
is intentionally light‑weight and focuses on the core algorithmic structure
rather than raw performance.

The training procedure alternates between self‑play data collection and neural
network fitting.  Two networks are used:

* ``Advantage Network`` approximates accumulated counterfactual regret for each
  information set/action pair.
* ``Strategy Network`` estimates the average strategy of the player.

In contrast to the DQN based approach used in :mod:`scripts.train_ai`, the
DeepCFR paradigm minimises regret instead of maximising expected reward.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from one_o_one.game import Action, Card, State, action_mask, reset, step

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_SAVE_DIR = PROJECT_ROOT / "models"

NUM_PLAYERS = 4
ACTION_SIZE = len(Action)
HISTORY_LENGTH = 48
STATE_SIZE = 35 + (NUM_PLAYERS + ACTION_SIZE) * HISTORY_LENGTH
LEARNING_RATE = 1e-3
MEMORY_CAPACITY = 10_000
BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# State representation utilities (copied from train_ai for consistency)
# ---------------------------------------------------------------------------


def _to_one_hot(num: int, max_val: int) -> list[float]:
    return [1.0 if i == num else 0.0 for i in range(max_val)]


def _rank_value(card: Card | None) -> int:
    return int(card.rank) if card is not None else 0


def encode_state(s: State) -> np.ndarray:
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

    return np.array(
        hand_vec + total_vec + dir_vec + penalty_vec + history_vec,
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Reservoir memory structures
# ---------------------------------------------------------------------------


class ReservoirMemory:
    """Simple reservoir sampling memory buffer."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data: list[tuple[Any, ...]] = []
        self.n_seen = 0

    def add(self, item: tuple[Any, ...]) -> None:
        self.n_seen += 1
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            idx = random.randint(0, self.n_seen - 1)
            if idx < self.capacity:
                self.data[idx] = item

    def sample(self, batch_size: int) -> list[tuple[Any, ...]]:
        if not self.data:
            return []
        return random.sample(self.data, min(batch_size, len(self.data)))


class AdvantageMemory(ReservoirMemory):
    """Stores (state, action, regret) tuples."""


class StrategyMemory(ReservoirMemory):
    """Stores (state, strategy) tuples."""


# ---------------------------------------------------------------------------
# Neural networks
# ---------------------------------------------------------------------------


class AdvantageNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden: int = 128):
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(x))


class StrategyNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden: int = 128):
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(x))


# ---------------------------------------------------------------------------
# DeepCFR agent
# ---------------------------------------------------------------------------


@dataclass
class DeepCFR:
    """Counterfactual regret minimisation agent with neural networks."""

    state_size: int = STATE_SIZE
    action_size: int = ACTION_SIZE
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self) -> None:
        logger.info("Using device: %s", self.device)
        self.adv_net = AdvantageNetwork(self.state_size, self.action_size).to(
            self.device
        )
        self.strat_net = StrategyNetwork(self.state_size, self.action_size).to(
            self.device
        )
        self.adv_optimizer = optim.Adam(self.adv_net.parameters(), lr=LEARNING_RATE)
        self.strat_optimizer = optim.Adam(self.strat_net.parameters(), lr=LEARNING_RATE)
        self.adv_memory = AdvantageMemory(MEMORY_CAPACITY)
        self.strat_memory = StrategyMemory(MEMORY_CAPACITY)

    # ------------------------------------------------------------------
    # Self-play and data collection
    # ------------------------------------------------------------------

    def _policy(self, state: State) -> tuple[Action, np.ndarray]:
        """Sample an action from the current strategy network."""
        state_vec = torch.tensor(encode_state(state), device=self.device)
        logits = self.strat_net(state_vec)
        mask = torch.tensor(action_mask(state), device=self.device, dtype=torch.float32)
        # Prevent illegal actions
        logits = logits + (mask - 1) * 1e9
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        action = np.random.choice(np.arange(self.action_size), p=probs)
        return Action(action), probs

    def play_game(self) -> None:
        """Run one episode of self-play collecting regret and strategy data."""
        state = reset(NUM_PLAYERS)
        trajectory: list[tuple[np.ndarray, int, np.ndarray]] = []
        while True:
            action, strat = self._policy(state)
            state_vec = encode_state(state)
            next_state, reward, done, _ = step(state, action)
            trajectory.append((state_vec, int(action), strat))
            if done:
                break
            state = next_state
        # Very small and naive regret estimate: difference between 1 and
        # probability of chosen action.  This is merely a placeholder to
        # demonstrate memory usage.
        for st_vec, act, strat in trajectory:
            regret_vec = np.zeros(self.action_size, dtype=np.float32)
            regret_vec[act] = 1.0 - strat[act]
            self.adv_memory.add((st_vec, act, regret_vec[act]))
            self.strat_memory.add((st_vec, strat))

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------

    def _train_advantage(self, epochs: int = 1) -> None:
        if not self.adv_memory.data:
            return
        self.adv_net = AdvantageNetwork(self.state_size, self.action_size).to(
            self.device
        )
        self.adv_optimizer = optim.Adam(self.adv_net.parameters(), lr=LEARNING_RATE)
        for _ in range(epochs):
            batch = self.adv_memory.sample(BATCH_SIZE)
            if not batch:
                continue
            states = torch.tensor([b[0] for b in batch], device=self.device)
            actions = torch.tensor([b[1] for b in batch], device=self.device)
            regrets = torch.tensor([b[2] for b in batch], device=self.device)
            self.adv_optimizer.zero_grad()
            output = self.adv_net(states)
            pred = output.gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = torch.mean((pred - regrets) ** 2)
            loss.backward()  # type: ignore[no-untyped-call]
            self.adv_optimizer.step()

    def _train_strategy(self, epochs: int = 1) -> None:
        if not self.strat_memory.data:
            return
        self.strat_net = StrategyNetwork(self.state_size, self.action_size).to(
            self.device
        )
        self.strat_optimizer = optim.Adam(self.strat_net.parameters(), lr=LEARNING_RATE)
        for _ in range(epochs):
            batch = self.strat_memory.sample(BATCH_SIZE)
            if not batch:
                continue
            states = torch.tensor([b[0] for b in batch], device=self.device)
            target = torch.tensor([b[1] for b in batch], device=self.device)
            self.strat_optimizer.zero_grad()
            logits = self.strat_net(states)
            log_probs = torch.log_softmax(logits, dim=-1)
            loss = -(target * log_probs).sum(dim=-1).mean()
            loss.backward()  # type: ignore[no-untyped-call]
            self.strat_optimizer.step()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, iterations: int = 1) -> None:
        """Run a number of DeepCFR iterations."""
        for i in range(iterations):
            logger.info("Starting iteration %s", i + 1)
            self.play_game()
            # After gathering data the networks are retrained from scratch
            self._train_advantage()
            self._train_strategy()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    agent = DeepCFR()
    agent.train(iterations=1)
    MODEL_SAVE_DIR.mkdir(exist_ok=True)
    torch.save(agent.strat_net.state_dict(), MODEL_SAVE_DIR / "strategy_net.pt")
    torch.save(agent.adv_net.state_dict(), MODEL_SAVE_DIR / "advantage_net.pt")


if __name__ == "__main__":
    main()
