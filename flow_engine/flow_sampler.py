from __future__ import annotations



from dataclasses import dataclass

import math

from typing import Optional



from .flow_runtime import ModelState

from .flow_selector import SelectionStrategy





@dataclass

class SamplerDecision:

    strategy: SelectionStrategy

    explore_ratio: float

    reason: str





@dataclass

class AdaptiveSampler:

    low_conf_threshold: float = 0.6

    mid_conf_threshold: float = 0.8

    strategy_space: tuple[SelectionStrategy, ...] = (

        "uncertainty_first",

        "hard_focus",

    )

    exploration_strength: float = 0.8



    def decide(self, model_state: ModelState) -> SamplerDecision:



        if model_state.avg_confidence < self.low_conf_threshold:

            return SamplerDecision(

                strategy="uncertainty_first",

                explore_ratio=0.55,

                reason="Low confidence phase; prioritize high-loss uncertain samples with hard-data coverage.",

            )

        if model_state.avg_confidence < self.mid_conf_threshold:

            return SamplerDecision(

                strategy="uncertainty_first",

                explore_ratio=0.35,

                reason="Mid confidence phase; keep importance-first uncertainty schedule.",

            )

        return SamplerDecision(

            strategy="hard_focus",

            explore_ratio=0.2,

            reason="High confidence phase; focus on hardest samples.",

        )



    def _bandit_decide(self, model_state: ModelState) -> Optional[SamplerDecision]:

        tried = sum(model_state.strategy_counts.get(s, 0) for s in self.strategy_space)

        if tried < len(self.strategy_space):

            for strategy in self.strategy_space:

                if model_state.strategy_counts.get(strategy, 0) == 0:

                    return SamplerDecision(

                        strategy=strategy,

                        explore_ratio=1.0,

                        reason=f"Bandit warmup: probing strategy '{strategy}'.",

                    )

        if tried < 8:

            return None

        total = max(1, tried)

        best_strategy = self.strategy_space[0]

        best_score = -1e9

        for strategy in self.strategy_space:

            count = max(1, model_state.strategy_counts.get(strategy, 0))

            reward = model_state.strategy_rewards.get(strategy, 0.0)

            ucb = reward + self.exploration_strength * math.sqrt(math.log(total + 1) / count)

            if ucb > best_score:

                best_score = ucb

                best_strategy = strategy

        return SamplerDecision(

            strategy=best_strategy,

            explore_ratio=0.35,

            reason=f"Bandit-UCB selected '{best_strategy}' (score={best_score:.3f}).",

        )

