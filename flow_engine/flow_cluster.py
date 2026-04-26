from __future__ import annotations



from dataclasses import dataclass, field



from .flow_core import EncodedSample





@dataclass

class OnlineClusterBackend:

    """Lightweight online centroid clustering for selection policies."""



    n_clusters: int = 8

    centroids: list[list[float]] = field(default_factory=list)

    counts: list[int] = field(default_factory=list)



    def assign(self, sample: EncodedSample, update: bool = True) -> int:

        if not sample.vector:

            return 0

        if not self.centroids:
            self._initialize(sample.vector)
            return 0

        cluster = self._closest_centroid(sample.vector)
        if update:
            self._update_centroid(cluster, sample.vector)

        return cluster



    def batch_assign(self, samples: list[EncodedSample], update: bool = True) -> dict[str, int]:
        if not self.centroids and samples:
            seed_vector = next((sample.vector for sample in samples if sample.vector), [])
            if seed_vector:
                self._initialize(seed_vector)
        return {sample.sample_id: self.assign(sample, update=update) for sample in samples}

    def _initialize(self, vector: list[float]) -> None:
        self.centroids = [list(vector) for _ in range(self.n_clusters)]
        self.counts = [1 for _ in range(self.n_clusters)]



    def _closest_centroid(self, vector: list[float]) -> int:

        best_idx = 0

        best_dist = float("inf")

        for idx, centroid in enumerate(self.centroids):

            width = min(len(vector), len(centroid))

            if width == 0:

                continue

            dist = sum((vector[i] - centroid[i]) ** 2 for i in range(width)) / width

            if dist < best_dist:

                best_dist = dist

                best_idx = idx

        return best_idx



    def _update_centroid(self, cluster: int, vector: list[float]) -> None:

        centroid = self.centroids[cluster]

        width = min(len(vector), len(centroid))

        self.counts[cluster] += 1

        alpha = 1.0 / self.counts[cluster]

        for idx in range(width):

            centroid[idx] = centroid[idx] * (1.0 - alpha) + vector[idx] * alpha

