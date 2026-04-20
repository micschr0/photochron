## ADDED Requirements

### Requirement: CLI command structure
The CLI SHALL implement all commands defined in `agent_docs/commands.md` using the Typer framework.

#### Scenario: Command availability
- **WHEN** running `python -m photochron --help`
- **THEN** the output SHALL list the commands: `run`, `cluster`, `rerun`, `status`
- **AND** each command SHALL have a brief description matching the architecture docs

#### Scenario: Command help text
- **WHEN** running `python -m photochron run --help`
- **THEN** the output SHALL describe the `--input` and `--output` parameters
- **AND** it SHALL mention the `--dry-run` option

### Requirement: Rich terminal output
The CLI SHALL use Rich for progress reporting and formatted output.

#### Scenario: Progress reporting
- **WHEN** running the pipeline on multiple photos
- **THEN** there SHALL be a progress bar showing completion percentage
- **AND** stage names SHALL be displayed during execution

#### Scenario: Formatted output
- **WHEN** running `photochron status`
- **THEN** the output SHALL be formatted as a table with columns for stage name, status, and photo count
- **AND** low-confidence results SHALL be highlighted in yellow

### Requirement: Parameter validation
The CLI SHALL validate input parameters and provide helpful error messages.

#### Scenario: Invalid input directory
- **WHEN** running `photochron run --input /nonexistent/path`
- **THEN** the CLI SHALL exit with a non-zero exit code
- **AND** display an error message indicating the directory does not exist

#### Scenario: Missing required parameters
- **WHEN** running `photochron run` without `--input`
- **THEN** the CLI SHALL show the help text and indicate missing required arguments