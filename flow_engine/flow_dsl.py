from __future__ import annotations



from dataclasses import dataclass, field

from typing import Any, Callable, Dict



from .flow_core import FlowSample





Predicate = Callable[[FlowSample], bool]

Action = Callable[[FlowSample], Dict[str, Any]]





@dataclass

class Rule:

    name: str

    when: Predicate

    then: Action

    priority: int = 100





@dataclass

class RuleEngine:

    rules: list[Rule] = field(default_factory=list)



    def add_rule(self, rule: Rule) -> None:

        self.rules.append(rule)

        self.rules.sort(key=lambda item: item.priority)



    def evaluate(self, sample: FlowSample) -> list[dict[str, Any]]:

        results: list[dict[str, Any]] = []

        for rule in self.rules:

            if rule.when(sample):

                payload = dict(rule.then(sample))

                payload["rule"] = rule.name

                payload["priority"] = rule.priority

                results.append(payload)

        return results





def example_rules() -> RuleEngine:

    """
    Example 1: Uncertain-heavy sample should be sampled first.
    Example 2: Sudden volatility jump should force curriculum boost.
    """



    engine = RuleEngine()



    engine.add_rule(

        Rule(

            name="uncertainty_first",

            priority=10,

            when=lambda sample: float(sample.metadata.get("uncertainty", 0.0)) > 0.6,

            then=lambda sample: {"strategy": "uncertainty_first", "boost": 1.25},

        )

    )



    engine.add_rule(

        Rule(

            name="volatility_boost",

            priority=20,

            when=lambda sample: float(sample.metadata.get("volatility_jump", 0.0)) > 0.4,

            then=lambda sample: {"strategy": "volatility_first", "boost": 1.15},

        )

    )



    return engine

