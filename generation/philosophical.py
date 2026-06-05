

"""Pseudo-profound quote generator.

Python port of a Chopra-style quote generator using typed phrase banks and a
small probability of generating a simple animal statement.

From "The Pseudo-Profound Art of Random Sentence Generation" by Gordon Pennycook, James Allan Cheyne, Nathaniel Barr, Derek J. Koehler, and Jonathan A. Fugelsang (2015).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass
class PseudoProfoundGenerator:
    """Generate fake Deepak Chopra-style quotes."""

    rng: random.Random = field(default_factory=random.Random)
    phrase_banks: List[List[str]] = field(default_factory=lambda: [
        [
            "The universe",
            "Your consciousness",
            "Quantum physics",
            "The unpredictable",
            "The unexplainable",
            "Our consciousness",
            "The soul",
            "Eternal stillness",
            "The cosmos",
            "Your desire",
            "Intuition",
            "Imagination",
            "Orderliness",
            "Wholeness",
            "The invisible",
            "Your body",
            "Awareness",
            "Perception",
            "God",
            "Knowledge",
            "Greatness",
            "Nature",
            "Your movement",
            "Everything",
            "Freedom",
            "Infinity",
            "Culture",
            "Perceptual reality",
            "Evolution",
            "Existence",
            "The ego",
            "Interdependence",
            "The world",
            "The mind",
            "Hidden meaning",
            "Love",
            "Good health",
            "The future",
            "Self power",
            "The web of life",
            "Your heart",
            "The secret of the universe",
            "Information",
            "Death",
            "Each of us",
            "Emotional intelligence",
            "Experiential truth",
            "The Higgs boson",
            "Qualia",
            "The future",
            "Non-judgment",
            "The human nervous system",
            "The physical world",
            "Making tea",
            "The key to joy",
            "Innocence",
        ],
        [
            "relies on",
            "depends on",
            "embraces",
            "requires",
            "illuminates",
            "is the ground of",
            "creates",
            "inspires",
            "nurtures",
            "heals",
            "gives rise to",
            "is rooted in",
            "arises and subsides in",
            "projects onto",
            "explores",
            "is the wisdom of",
            "is inherent in",
            "is the path to",
            "experiences",
            "comprehends",
            "explains",
            "is beyond",
            "transcends",
            "is the continuity of",
            "regulates",
            "meditates on",
            "serves",
            "is inside",
            "is in the midst of",
            "shapes",
            "transforms",
            "undertakes",
            "fascinates",
            "influences",
            "expresses",
            "opens",
            "is reborn in",
            "is an ingredient of",
            "unfolds into",
            "constructs",
            "differentiates into",
            "is only possible in",
            "grows through",
            "exists as",
            "reflects",
            "belongs to",
            "quiets",
            "imparts reality to",
            "is the foundation of",
            "alleviates",
            "is at the heart of",
            "compliments",
            "corresponds to",
            "results from",
        ],
        [
            "your own",
            "infinite",
            "self-righteous",
            "unbridled",
            "cosmic",
            "unique",
            "visible",
            "great",
            "boundless",
            "quantum",
            "an abundance of",
            "subtle",
            "humble",
            "universal",
            "an expression of",
            "intrinsic",
            "ephemeral",
            "total",
            "reckless",
            "pure",
            "positive",
            "the expansion of",
            "the mechanics of",
            "the doorway to",
            "a symphony of",
            "irrational",
            "essential",
            "spontaneous",
            "karmic",
            "deep",
            "unparalleled",
            "incredible",
            "the flow of",
            "mortal",
            "potential",
            "the barrier of",
            "exponential",
            "descriptions of",
            "intricate",
            "new",
            "existential",
            "the light of",
            "precious",
            "subjective",
            "immortal",
            "species specific",
            "a jumble of",
            "dimensionless",
            "the progressive expansion of",
            "formless",
            "total acceptance of",
            "innumerable",
        ],
        [
            "joy",
            "creativity",
            "life",
            "possibilities",
            "sensations",
            "experiences",
            "energy",
            "happiness",
            "reality",
            "knowledge",
            "facts",
            "space time events",
            "opportunities",
            "sexual energy",
            "chaos",
            "truth",
            "destiny",
            "success",
            "choices",
            "acceptance",
            "silence",
            "positivity",
            "excellence",
            "belonging",
            "abstract beauty",
            "balance",
            "fulfillment",
            "bliss",
            "actions",
            "potentiality",
            "mysteries",
            "marvel",
            "external reality",
            "self-knowledge",
            "photons",
            "mortality",
            "timelessness",
            "force fields",
            "brightness",
            "neural networks",
            "human observation",
            "love",
            "boundaries",
            "brains",
            "phenomena",
            "miracles",
            "observations",
        ],
    ])
    animal_subjects: List[str] = field(default_factory=lambda: [
        "Dogs",
        "Cats",
        "Mice",
        "Rats",
        "Elephants",
        "Kittens",
    ])
    animal_predicates: List[str] = field(default_factory=lambda: [
        "insects",
        "horses",
        "llamas",
        "mules",
        "birds",
        "sterile",
        "plants",
    ])
    animal_quote_percent: int = 2

    def _sample_without_replacement(self, items: Sequence[str]) -> str:
        if not items:
            raise ValueError("Cannot sample from an empty phrase bank.")
        return self.rng.choice(list(items))

    def generate_quote(self) -> str:
        """Generate a single quote."""
        if self.rng.randrange(100) < self.animal_quote_percent:
            subject = self._sample_without_replacement(self.animal_subjects)
            predicate = self._sample_without_replacement(self.animal_predicates)
            return f"{subject} are {predicate}"

        parts = [self._sample_without_replacement(bank) for bank in self.phrase_banks]
        return " ".join(parts)

    def generate(self, n: int = 1) -> List[str]:
        """Generate `n` quotes."""
        if n < 1:
            raise ValueError("n must be at least 1.")
        return [self.generate_quote() for _ in range(n)]


DEFAULT_GENERATOR = PseudoProfoundGenerator()


def generate_pseudo_profound_quotes(n: int = 10, seed: int | None = None) -> List[str]:
    """Convenience function for one-off generation."""
    rng = random.Random(seed)
    generator = PseudoProfoundGenerator(rng=rng)
    return generator.generate(n=n)


if __name__ == "__main__":
    examples = generate_pseudo_profound_quotes(n=10, seed=42)
    for i, quote in enumerate(examples, start=1):
        print(f"{i}. {quote}")