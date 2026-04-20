## ADDED Requirements

### Requirement: Python package structure
The project SHALL use a `src/photochron/` package layout with clear module separation reflecting the 6-stage pipeline architecture.

#### Scenario: Package layout validation
- **WHEN** inspecting the project root directory
- **THEN** there SHALL be a `src/photochron/` directory containing the main package
- **AND** there SHALL be subdirectories: `cli/`, `pipeline/`, `store/`, `config/`, `models/`, `utils/`

#### Scenario: Module imports
- **WHEN** importing the photochron package
- **THEN** modules SHALL be importable via `from photochron.cli import main`
- **AND** there SHALL be no top-level Python files outside `src/` except entry points

### Requirement: Development setup
The project SHALL support editable installation for development and have proper tool configuration.

#### Scenario: Editable installation
- **WHEN** running `pip install -e .` from project root
- **THEN** the photochron package SHALL be importable in the Python environment
- **AND** changes to source code SHALL be immediately available without reinstallation

#### Scenario: Tool configuration
- **WHEN** checking project root
- **THEN** there SHALL be a `pyproject.toml` file with `[project]` section
- **AND** there SHALL be configuration for linting (`ruff`), type checking (`mypy`), and testing (`pytest`)