"""Gold-firewalled FullWiki task loader over the pinned HotpotQA snapshot.

The original distractor snapshot supplies question IDs, answers, and the official
sentence segmentation. Agents receive only ``PublicQuestion``. Retrieval is
against the external BEIR Wikipedia paragraph index, not the ten distractor
paragraphs stored with the source record.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from hotpot import loader as source


@dataclass(frozen=True)
class PublicQuestion:
    id: str
    question: str
    level: str = ""
    qtype: str = ""


@dataclass(frozen=True)
class RuntimeExample(PublicQuestion):
    canonical_context: list[tuple[str, list[str]]] = field(default_factory=list)

    def public(self) -> PublicQuestion:
        return PublicQuestion(self.id, self.question, self.level, self.qtype)

    def sentences_for_title(self, title: str) -> list[str] | None:
        for candidate, sentences in self.canonical_context:
            if candidate == title:
                return list(sentences)
        return None


@dataclass(frozen=True)
class ScorerExample(RuntimeExample):
    answer: str = ""
    supporting_facts: list[tuple[str, int]] = field(default_factory=list)

    def runtime(self) -> RuntimeExample:
        return RuntimeExample(self.id, self.question, self.level, self.qtype,
                              [(title, list(sentences))
                               for title, sentences in self.canonical_context])


def from_source(example: source.Example) -> ScorerExample:
    return ScorerExample(
        id=example.id,
        question=example.question,
        level=example.level,
        qtype=example.qtype,
        canonical_context=[(title, list(sentences)) for title, sentences in example.context],
        answer=example.answer,
        supporting_facts=list(example.supporting_facts),
    )


def open_snapshot():
    return source.open_pinned_snapshot()


def public_manifest_rows(snapshot=None) -> list[dict[str, str]]:
    return source.public_manifest_rows(snapshot or open_snapshot())


def load_examples(rows: Iterable[dict[str, str]], snapshot=None) -> list[ScorerExample]:
    snap = snapshot or open_snapshot()
    return [from_source(item) for item in source.load_examples(snap, rows)]
