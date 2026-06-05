

"""Corporate bullshit generator.

Inspired by the structure described by Littrell et al. and the New Age Bullshit
Generator design pattern: sentence skeletons plus typed word banks.

This module generates syntactically coherent, semantically vacuous corporate
statements by filling hand-written skeletons with randomly sampled jargon.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence


TOKEN_PATTERN = re.compile(r"\[([A-Za-z0-9_]+)\]")


@dataclass(frozen=True)
class Template:
    """A sentence skeleton with bracketed placeholders."""

    text: str


WORD_BANKS: Dict[str, List[str]] = {
    # Adjectives
    "vAdj": [
        "adaptive",
        "agile",
        "augmented",
        "integrated",
        "innovative",
        "intelligent",
        "end-to-end",
        "exponential",
        "iterative",
        "energized",
    ],
    "cAdj": [
        "balanced",
        "bleeding-edge",
        "business-focused",
        "collaborative",
        "cross-functional",
        "customer-centered",
        "data-driven",
        "dynamic",
        "frictionless",
        "growth-based",
        "hyper-connected",
        "mission-critical",
        "performance-focused",
        "process-facing",
        "renewed",
        "resonating",
        "rigorous",
        "scalable",
        "strategic",
        "synergistic",
        "transformational",
        "value-centered",
        "visionary",
    ],
    "NounAdj": [
        "key",
        "strategic",
        "customer-facing",
        "high-impact",
        "human-centered",
        "mission-aligned",
        "value-added",
        "future-ready",
        "cross-back",
        "results-driven",
    ],
    # Nouns / noun phrases
    "vNoun": [
        "adaptive coherence",
        "augmented business visualization",
        "end-state vision",
        "empowerment",
        "ecosystem",
        "innovation engine",
        "operating ethos",
        "outside-the-box alignment",
        "upstream momentum",
        "AI-based platform",
    ],
    "cNoun": [
        "balanced scorecard",
        "back-end architecture",
        "bandwidth",
        "benchmark",
        "brand experience",
        "competitive heat",
        "core values",
        "culture fit",
        "customer journey",
        "flow chart",
        "framework",
        "growth mindset",
        "heritage",
        "marketplace",
        "pain point",
        "pipeline",
        "strategic intent",
        "success story",
        "thought leadership",
        "value chain",
        "vector",
        "wheelhouse",
    ],
    "nNoun": [
        "acquisition",
        "action plan",
        "analytics",
        "architecture",
        "brand champion",
        "business model",
        "change driver",
        "company",
        "conversation",
        "customer",
        "future",
        "growth vector",
        "human spirit",
        "market",
        "mission",
        "opportunity",
        "platform",
        "portfolio",
        "process",
        "resource",
        "strategy",
        "success",
        "transmission",
    ],
    "nNounPlural": [
        "analytics",
        "best practices",
        "brand champions",
        "capabilities",
        "change drivers",
        "conversations",
        "core values",
        "frameworks",
        "integrated networks",
        "key deliverables",
        "key learnings",
        "key takeaways",
        "milestones",
        "pain points",
        "people solutions",
        "performance vectors",
        "predictors",
        "services",
        "strategic initiatives",
        "swim lanes",
    ],
    "nProduct": [
        "balanced scorecard",
        "business platform",
        "customer experience",
        "digital ecosystem",
        "enterprise solution",
        "innovation roadmap",
        "operating model",
        "service portfolio",
        "strategic framework",
        "value creation",
    ],
    "rolePlural": [
        "global partners",
        "leaders",
        "stakeholders",
        "team members",
        "thought partners",
        "vision custodians",
    ],
    "place": [
        "ecosystem",
        "industry",
        "market",
        "marketplace",
        "organization",
        "space",
        "world",
    ],
    # Verbs
    "verb": [
        "activate",
        "actualize",
        "architect",
        "benchmark",
        "coach",
        "concretize",
        "engage",
        "excel",
        "foster",
        "grasp",
        "growth-hack",
        "ideate",
        "leverage",
        "maximize",
        "optimize",
        "orchestrate",
        "pivot",
        "potentiate",
        "pressure-test",
        "reimagine",
        "solution",
        "sunset",
    ],
    "ingVerb": [
        "actualizing",
        "architecting",
        "benchmarking",
        "building bridges to success",
        "cheerleading",
        "coaching",
        "delivering",
        "downloading",
        "drilling down one more click on",
        "driving",
        "engaging",
        "executing",
        "growth hacking",
        "joining with",
        "leveraging",
        "moving forward with",
        "pivoting",
        "pressure-testing",
        "solutioning",
        "standing on the shoulders of giants",
    ],
    "pastVerb": [
        "actualized",
        "architected",
        "benchmarked",
        "concretized",
        "engaged",
        "fostered",
        "leveraged",
        "optimized",
        "reimagined",
        "sunsetted",
    ],
    # Connective / style phrases
    "advPhrase": [
        "as effectively as we can",
        "at scale",
        "going forward",
        "in a disciplined way",
        "in real time",
        "like no other company anywhere in the world",
        "with intention",
        "with every ounce of our beings",
    ],
    "beliefPhrase": [
        "firmly believe",
        "know",
        "recognize",
        "remain confident",
        "see a clear line of sight",
    ],
    "timePhrase": [
        "Each day",
        "In this moment",
        "Moving forward",
        "Over the next fiscal year",
        "With each fiscal year",
    ],
}


TEMPLATES: List[Template] = [
    Template(
        "Our [vNoun] is to [verb] our [NounAdj] [nNounPlural] by [ingVerb] our efforts on [ingVerb] the [cAdj] [vNoun] of our [nProduct]."
    ),
    Template(
        "By [ingVerb] our [nNounPlural], we will [verb] a [cAdj] level of [vNoun] and [cNoun] in the [place]."
    ),
    Template(
        "This [cAdj] look at our [cNoun] will ensure that we are [ingVerb] and [ingVerb] our [nNounPlural] [advPhrase] in order to [verb] our [vNoun]."
    ),
    Template(
        "We [beliefPhrase] that [cAdj] magic happens when you [verb] [NounAdj] [nNounPlural]."
    ),
    Template(
        "Working at the intersection of [vNoun] and [cNoun], we will [verb] a [cAdj] level of [vNoun] in a world defined by [ingVerb] to [verb] on a [cAdj] [place]."
    ),
    Template(
        "[timePhrase], we help our [nNounPlural] thrive in a revolution of [nNounPlural] fueled by our [vNoun]."
    ),
    Template(
        "With each fiscal year, we [verb] with every ounce of our beings by [ingVerb] our [nNoun], [nNoun], and [nNounPlural]."
    ),
    Template(
        "By solving the [cNoun] of [nNounPlural] with our [nNounPlural], we will [verb] a [cAdj] level of [vNoun] and [cNoun] in the market between us and others who are [ingVerb] to [verb] on a similar [nProduct]."
    ),
    Template(
        "We will [verb] our [nNounPlural] by joining with our [cAdj], [cAdj] [rolePlural] to better [verb] our [cNoun]."
    ),
    Template(
        "We will [verb] our [nNounPlural] in delivering [cAdj], [cAdj] [nNounPlural] like no other company anywhere in the world."
    ),
    Template(
        "Pivoting on our [cNoun] will be the [cAdj] view in how we [verb] [cAdj] outcomes."
    ),
    Template(
        "As an emerging leader grounded in a mission to [verb] and nurture the human spirit, we have always aspired to make [cAdj] [nNounPlural], [ingVerb] people and communities around the world."
    ),
    Template(
        "Our [cNoun] comes from the [cAdj] [ingVerb] of several new [cAdj], [nNounPlural] that capitalize on our heritage to [verb] our future when [ingVerb]."
    ),
    Template(
        "Focused on [vNoun] in a [cAdj] and [vAdj] world, we provide [vNoun] and [cAdj] [nNounPlural] for [ingVerb] through a [vAdj] [nProduct]."
    ),
    Template(
        "Our goal is to [verb] our [nNounPlural] by focusing our efforts on [ingVerb] the current [cNoun] of our [vNoun], driving an [vAdj] [cNoun] with our [nNounPlural], and [ingVerb] [cAdj] [nNounPlural] to our [vNoun]."
    ),
    Template(
        "Standing on the shoulders of giants, our [nNoun] comes from the [cAdj], [cAdj] thinking of several new [cAdj] [nNounPlural] that capitalize on our heritage to [verb] our successful future."
    ),
    Template(
        "In order to [verb] our [nNoun], we must continually [verb] and review every part of our [nNoun] operations."
    ),
    Template(
        "You have to appreciate that the [nNounPlural] we have set in these [nNounPlural] provide a roadmap for this [cNoun]."
    ),
    Template(
        "As brands build out a [place] footprint, they look for the [cAdj] global POV that has always been part of our [cNoun]."
    ),
    Template(
        "In the [cAdj] channel, our company [pastVerb] [nNounPlural] not aligned with [cNoun] and invested in presentation through both [cAdj] assortments and dedicated, customized real estate in key [nNounPlural]."
    ),
    Template(
        "We actually think that the industry is at a place where you can see line of sight to the [cNoun] just fundamentally changing in a very short period of time."
    ),
    Template(
        "This [cAdj] look at our company will ensure that we are [ingVerb] and [ingVerb] [cAdj] impact with our [nNounPlural] [advPhrase] in order to fundamentally disrupt our [place]."
    ),

]


REAL_CORPORATE_STATEMENTS: List[str] = [
    "In order to reinvigorate our company, we must continually analyze and review every part of our company operations.",
    "This rigorous look at our business will ensure that we are managing and optimizing our resources as effectively as we can in order to improve the brand experience.",
    "Our success comes from the rigorous execution of several new strategic initiatives that capitalize on our heritage to drive our successful future.",
    "By focusing again on the customer experience, we will create a renewed level of meaningful differentiation and separation in the market between us and our competitors.",
    "We will leverage our extensive business networks, market knowledge, and logistical expertise to produce high-value, bundled products for an increasing number of global customers.",
    "We have robust networks of strategic assets that we own or have contractual access to, which give us greater flexibility and speed to reliably deliver widespread logistical solutions.",
    "Under design or construction in iconic, global cities, we will join our roasteries in delivering an immersive, ultra-premium, coffee-forward experience like none other anywhere in the world.",
    "We plan to right-size our manufacturing operations to align to the new strategy and take advantage of integration opportunities.",
    "The evolving social and digital media platforms and highly innovative and relevant payment capabilities are causing seismic changes in consumer behavior and creating equally disruptive opportunities for business.",
    "In this ever-changing society, the most powerful and enduring brands are built from the heart. Their foundations are stronger because they are built with the strength of the human spirit, not an ad campaign. The companies that are lasting are those that are authentic.",
    "Because of our iconic brands, our unending commitment to premium content, and the innovation of our teams, we have permission from the market to be a world-class, tier-one partner.",
    "Our performance and capabilities cannot be compared to our peers. We have a proven business concept that is eminently scalable in our existing businesses and adaptable enough to extend to new markets.",
]



class CorporateBullshitGenerator:
    """Generate corporate-sounding pseudo-profound statements."""

    def __init__(
        self,
        templates: Sequence[Template] | None = None,
        word_banks: Dict[str, Sequence[str]] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.templates: List[Template] = list(templates or TEMPLATES)
        raw_banks = word_banks or WORD_BANKS
        self.word_banks: Dict[str, List[str]] = {
            key: list(values) for key, values in raw_banks.items()
        }
        self.rng = rng or random.Random()
        self._validate_templates()

    def _validate_templates(self) -> None:
        for template in self.templates:
            placeholders = TOKEN_PATTERN.findall(template.text)
            missing = [token for token in placeholders if token not in self.word_banks]
            if missing:
                joined = ", ".join(sorted(set(missing)))
                raise ValueError(
                    f"Template has placeholders with no matching word bank: {joined}\n"
                    f"Template: {template.text}"
                )

    def sample_word(self, category: str) -> str:
        words = self.word_banks[category]
        if not words:
            raise ValueError(f"Word bank '{category}' is empty.")
        return self.rng.choice(words)

    def render_template(self, template: Template) -> str:
        def replacer(match: re.Match[str]) -> str:
            category = match.group(1)
            return self.sample_word(category)

        text = TOKEN_PATTERN.sub(replacer, template.text)
        return self._clean_text(text)

    def generate(self, n: int = 1, unique: bool = False) -> List[str]:
        if n < 1:
            raise ValueError("n must be at least 1.")

        results: List[str] = []
        seen = set()

        while len(results) < n:
            template = self.rng.choice(self.templates)
            statement = self.render_template(template)
            if unique:
                if statement in seen:
                    continue
                seen.add(statement)
            results.append(statement)

        return results

    def generate_real_statement(self) -> str:
        """Return a real corporate statement from Littrell et al. Study 2."""
        return self.rng.choice(REAL_CORPORATE_STATEMENTS)

    def generate_real_statements(self, n: int = 1, unique: bool = False) -> List[str]:
        """Return `n` real corporate statements."""
        if n < 1:
            raise ValueError("n must be at least 1.")

        if unique:
            if n > len(REAL_CORPORATE_STATEMENTS):
                raise ValueError(
                    "Cannot generate more unique real statements than are available."
                )
            return self.rng.sample(REAL_CORPORATE_STATEMENTS, k=n)

        return [self.generate_real_statement() for _ in range(n)]

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        return text


DEFAULT_GENERATOR = CorporateBullshitGenerator()



def generate_corporate_bullshit(
    n: int = 10,
    seed: int | None = None,
    unique: bool = False,
) -> List[str]:
    """Convenience function for one-off generation."""
    rng = random.Random(seed)
    generator = CorporateBullshitGenerator(rng=rng)
    return generator.generate(n=n, unique=unique)


def generate_real_corporate_statements(
    n: int = 10,
    seed: int | None = None,
    unique: bool = False,
) -> List[str]:
    """Convenience function for sampling real corporate statements."""
    rng = random.Random(seed)
    generator = CorporateBullshitGenerator(rng=rng)
    return generator.generate_real_statements(n=n, unique=unique)


if __name__ == "__main__":
    print("Generated corporate bullshit:\n")
    examples = generate_corporate_bullshit(n=5, seed=42, unique=True)
    for i, statement in enumerate(examples, start=1):
        print(f"{i}. {statement}")

    print("\nReal corporate statements:\n")
    real_examples = generate_real_corporate_statements(n=5, seed=42, unique=True)
    for i, statement in enumerate(real_examples, start=1):
        print(f"{i}. {statement}")