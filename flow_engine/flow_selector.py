from __future__ import annotations



from dataclasses import dataclass, field

from typing import Optional
import math



from .flow_cluster import OnlineClusterBackend

from .flow_core import EncodedSample

from .flow_difficulty import DifficultyEstimator, DifficultyScore

from .flow_runtime import ModelState





SelectionStrategy = str





@dataclass

class SelectionItem:

    sample: EncodedSample

    difficulty: DifficultyScore





@dataclass

class FlowSelector:

    estimator: DifficultyEstimator

    default_strategy: SelectionStrategy = "uncertainty_first"

    diversity_weight: float = 0.2

    temporal_weight: float = 0.2

    cluster_backend: OnlineClusterBackend = field(default_factory=OnlineClusterBackend)



    def rank(

        self,

        encoded_samples: list[EncodedSample],

        model_state: ModelState,

        strategy: Optional[SelectionStrategy] = None,

    ) -> list[SelectionItem]:

        ranked, _cluster_map = self._rank_with_clusters(encoded_samples, model_state, strategy)
        return ranked

    def _rank_with_clusters(
        self,
        encoded_samples: list[EncodedSample],
        model_state: ModelState,
        strategy: Optional[SelectionStrategy] = None,
    ) -> tuple[list[SelectionItem], dict[str, int]]:
        use_strategy = strategy or self.default_strategy
        cluster_map = self.cluster_backend.batch_assign(encoded_samples, update=False)
        items = [SelectionItem(sample=s, difficulty=self.estimator.score(s, model_state)) for s in encoded_samples]



        if use_strategy == "easy_to_hard":

            phase = model_state.phase()

            reverse = phase == "stabilize"

            return sorted(items, key=lambda item: item.difficulty.total, reverse=reverse), cluster_map
        if use_strategy == "hard_focus":
            return sorted(
                items,
                key=lambda item: (item.difficulty.loss, item.difficulty.total),
                reverse=True,
            ), cluster_map

        if use_strategy == "uncertainty_first":

            return sorted(
                items,
                key=lambda item: self._importance_score(item, model_state, cluster_map[item.sample.sample_id]),
                reverse=True,
            ), cluster_map

        if use_strategy == "volatility_first":

            return sorted(items, key=lambda item: item.difficulty.volatility, reverse=True), cluster_map

        if use_strategy == "temporal_flow":

            return sorted(

                items,

                key=lambda item: self._temporal_score(item, model_state, cluster_map[item.sample.sample_id]),

                reverse=True,

            ), cluster_map

        return sorted(

            items,

            key=lambda item: self._mixed_objective(item, model_state, cluster_map[item.sample.sample_id]),

            reverse=True,
        ), cluster_map



    def select(

        self,

        encoded_samples: list[EncodedSample],

        model_state: ModelState,

        batch_size: int,

        strategy: Optional[SelectionStrategy] = None,

    ) -> list[SelectionItem]:

        ranked, cluster_map = self._rank_with_clusters(encoded_samples, model_state, strategy=strategy)
        use_strategy = strategy or self.default_strategy
        if use_strategy == "uncertainty_first":
            pool_size = max(1, int(0.3 * len(ranked)))
            pool = ranked[:pool_size]
            return self._diverse_topk(pool, max(1, batch_size), cluster_map)
        return self._diverse_topk(ranked, max(1, batch_size), cluster_map)

    def _importance_score(self, item: SelectionItem, model_state: ModelState, cluster_key: int) -> float:
        local_conf = model_state.cluster_confidence.get(cluster_key, model_state.avg_confidence)
        hard_bonus = max(0.0, 1.0 - local_conf)
        return (
            0.55 * item.difficulty.loss
            + 0.25 * item.difficulty.uncertainty
            + 0.10 * item.difficulty.error_history
            + 0.05 * item.difficulty.novelty
            + 0.05 * hard_bonus
        )



    def _mixed_objective(self, item: SelectionItem, model_state: ModelState, cluster_key: int) -> float:

        local_conf = model_state.cluster_confidence.get(cluster_key, model_state.avg_confidence)

        local_uncertainty = 1.0 - local_conf

        return (

            0.45 * item.difficulty.total

            + 0.25 * item.difficulty.uncertainty

            + 0.15 * item.difficulty.volatility

            + 0.15 * local_uncertainty

        )



    def _temporal_score(self, item: SelectionItem, model_state: ModelState, cluster_key: int) -> float:

        segment_count = item.sample.flow_features.get("segment_count", 0.0)

        event_count = item.sample.flow_features.get("event_count", 0.0)

        temporal = min(1.0, segment_count / max(1.0, event_count))

        local_conf = model_state.cluster_confidence.get(cluster_key, model_state.avg_confidence)

        return 0.5 * item.difficulty.total + 0.3 * temporal + 0.2 * (1.0 - local_conf)



    def _diverse_topk(self, ranked: list[SelectionItem], k: int, cluster_map: dict[str, int]) -> list[SelectionItem]:

        selected: list[SelectionItem] = []

        seen_clusters: set[int] = set()

        for item in ranked:

            cluster = cluster_map.get(item.sample.sample_id, 0)

            if cluster in seen_clusters and len(selected) < max(1, int(k * 0.7)):

                continue
            if any(self._cosine_similarity(item.sample.vector, chosen.sample.vector) > 0.95 for chosen in selected):
                continue

            selected.append(item)

            seen_clusters.add(cluster)

            if len(selected) >= k:

                return selected

        if len(selected) < k:

            for item in ranked:

                if item in selected:

                    continue

                selected.append(item)

                if len(selected) >= k:

                    break

        return selected

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        width = min(len(a), len(b))
        if width == 0:
            return 0.0
        dot = sum(a[i] * b[i] for i in range(width))
        norm_a = math.sqrt(sum(a[i] * a[i] for i in range(width)))
        norm_b = math.sqrt(sum(b[i] * b[i] for i in range(width)))
        if norm_a <= 1e-12 or norm_b <= 1e-12:
            return 0.0
        return dot / (norm_a * norm_b)



