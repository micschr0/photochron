"""Unit tests for ``PipelineRegistry.get_dependency_order`` (topological sort).

The pre-refactor implementation returned stages in registration order, which
silently produced wrong execution order when a stage was registered before its
declared dependency. The new implementation uses Kahn's algorithm with
registration order as the tiebreaker.
"""

from __future__ import annotations

import pytest

from photochron.pipeline import PipelineRegistry, PipelineStage


def _stage_cls(name: str, deps: list[str]) -> type[PipelineStage]:
    """Synthesise a minimal concrete PipelineStage class for use in tests."""

    class _S(PipelineStage):
        @property
        def name(self) -> str:
            return name

        @property
        def dependencies(self) -> list[str]:
            return deps

        def run(self, run_id: str, config_hash: str) -> None:  # pragma: no cover
            pass

    _S.__name__ = f"Stage_{name}"
    return _S


class TestTopologicalOrder:
    def test_independent_stages_keep_registration_order(self) -> None:
        reg = PipelineRegistry()
        reg.register(_stage_cls("a", []))
        reg.register(_stage_cls("b", []))
        reg.register(_stage_cls("c", []))
        assert reg.get_dependency_order() == ["a", "b", "c"]

    def test_dependency_pulls_predecessor_earlier(self) -> None:
        """Even if the dependent stage is registered first, it must run after its dep."""
        reg = PipelineRegistry()
        reg.register(_stage_cls("late", ["early"]))
        reg.register(_stage_cls("early", []))
        order = reg.get_dependency_order()
        assert order.index("early") < order.index("late")

    def test_full_six_stage_pipeline(self) -> None:
        """The real photochron stage shape: chain of six dependencies."""
        reg = PipelineRegistry()
        # Register out of order on purpose.
        reg.register(_stage_cls("output_layer", ["ranking_engine"]))
        reg.register(_stage_cls("ranking_engine", ["anchor_layer"]))
        reg.register(_stage_cls("anchor_layer", ["context_layer"]))
        reg.register(_stage_cls("context_layer", ["face_layer"]))
        reg.register(_stage_cls("face_layer", ["ingestion"]))
        reg.register(_stage_cls("ingestion", []))
        assert reg.get_dependency_order() == [
            "ingestion",
            "face_layer",
            "context_layer",
            "anchor_layer",
            "ranking_engine",
            "output_layer",
        ]

    def test_cycle_raises(self) -> None:
        reg = PipelineRegistry()
        reg.register(_stage_cls("a", ["b"]))
        reg.register(_stage_cls("b", ["a"]))
        with pytest.raises(RuntimeError, match="Cycle"):
            reg.get_dependency_order()

    def test_unknown_dependency_does_not_break_topo(self) -> None:
        """Unknown deps are surfaced by ``validate_dependencies``; topo skips them."""
        reg = PipelineRegistry()
        reg.register(_stage_cls("a", ["__missing__"]))
        order = reg.get_dependency_order()
        assert order == ["a"]
        assert reg.validate_dependencies() == [
            "Stage 'a' depends on unknown stage '__missing__'",
        ]
