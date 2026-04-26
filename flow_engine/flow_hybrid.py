from __future__ import annotations



from dataclasses import dataclass



from .flow_core import FlowSample

from .flow_dsl import RuleEngine

from .flow_trainer import FlowTrainer





@dataclass

class HybridDecision:

    strategy: str

    score: float

    source: str

    confidence: float





@dataclass

class HybridInferenceEngine:

    """
    Hybrid path:
    1) Rule engine creates deterministic scheduling hints
    2) Learned estimator calibrates with live model confidence
    """



    trainer: FlowTrainer

    rule_engine: RuleEngine

    rule_weight: float = 0.6

    learned_weight: float = 0.4



    def decide(self, sample: FlowSample) -> HybridDecision:

        encoded = self.trainer.encoder.encode(sample)

        learned_conf = self.trainer.model.predict_confidence(encoded)

        rule_results = self.rule_engine.evaluate(sample)



        if not rule_results:

            return HybridDecision(

                strategy="mixed",

                score=learned_conf,

                source="learned",

                confidence=learned_conf,

            )



        top_rule = rule_results[0]

        boost = float(top_rule.get("boost", 1.0))

        calibrated = min(1.0, self.rule_weight * boost + self.learned_weight * learned_conf)

        return HybridDecision(

            strategy=str(top_rule.get("strategy", "mixed")),

            score=calibrated,

            source=f"rule:{top_rule.get('rule', 'unknown')}",

            confidence=learned_conf,

        )

