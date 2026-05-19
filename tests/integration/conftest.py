"""Auto-mark every test under ``tests/integration/`` with the ``integration``
marker.

Plan items in the audit asked for individual files to declare
``pytestmark = pytest.mark.integration``. Doing it here in one place ensures
new contributors can drop a file under ``tests/integration/`` and it will be
correctly excluded from ``pytest -m "not integration"`` without having to
remember the boilerplate.
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        item.add_marker(pytest.mark.integration)
