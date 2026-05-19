"""Unit tests for the RunContext binding pattern.

Replaces the old "PipelineRunner mutates the Config singleton" pattern. The
runner now binds a :class:`RunContext` to each stage instance before calling
``run()``; stages read ``self.context.input_dir``, ``self.context.output_dir``,
and ``self.context.dry_run`` instead of mutating the global config.
"""

from __future__ import annotations

from pathlib import Path

from photochron.pipeline import PipelineStage, RunContext


class _NeedsContextStage(PipelineStage):
    @property
    def name(self) -> str:
        return "needs_ctx"

    @property
    def dependencies(self) -> list[str]:
        return []

    def run(self, run_id: str, config_hash: str) -> None:  # pragma: no cover
        pass


def test_unbound_context_is_none() -> None:
    """Default state — no context — is explicit, not silently coerced."""
    assert _NeedsContextStage().context is None


def test_bind_context_attaches_run_inputs(tmp_path: Path) -> None:
    stage = _NeedsContextStage()
    ctx = RunContext(
        run_id="r1",
        config_hash="h1",
        input_dir=tmp_path / "in",
        output_dir=tmp_path / "out",
        dry_run=True,
    )
    stage.bind_context(ctx)
    assert stage.context is ctx
    assert stage.context.input_dir == tmp_path / "in"
    assert stage.context.output_dir == tmp_path / "out"
    assert stage.context.dry_run is True


def test_context_is_immutable() -> None:
    """``RunContext`` is frozen — stages cannot rewrite shared state through it."""
    ctx = RunContext(run_id="r1", config_hash="h1")
    import dataclasses

    try:
        ctx.run_id = "tampered"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("RunContext must be frozen")
