## ADDED Requirements

### Requirement: Configuration file structure
The system SHALL use a `config.yaml` file with defaults from the architecture specification.

#### Scenario: Default configuration
- **WHEN** no custom configuration is provided
- **THEN** the system SHALL use default values matching the architecture spec
- **AND** these defaults SHALL be documented in the config file itself

#### Scenario: Configuration validation
- **WHEN** loading a configuration file with invalid values
- **THEN** the system SHALL raise a validation error with specific field information
- **AND** suggest corrected values where possible

### Requirement: Environment variable overrides
Configuration values SHALL be overrideable via environment variables.

#### Scenario: Environment precedence
- **WHEN** both config file and environment variable define a value
- **THEN** the environment variable SHALL take precedence
- **AND** the override SHALL be logged for debugging

#### Scenario: Environment variable naming
- **WHEN** setting environment variables
- **THEN** they SHALL follow the pattern `PHOTOCHRON_<SECTION>_<KEY>` in uppercase
- **AND** nested keys SHALL use underscores (e.g., `PHOTOCHRON_MODELS_INSIGHTFACE_VERSION`)

### Requirement: Anchors file format
The system SHALL support an `anchors.yaml` file for user-provided anchor data.

#### Scenario: Anchors file parsing
- **WHEN** providing a valid `anchors.yaml` file
- **THEN** the system SHALL parse persons, events, and known_dates sections
- **AND** validate date formats and constraint types

#### Scenario: Anchors file template
- **WHEN** no anchors file exists
- **THEN** the system SHALL create a template `anchors.yaml` with commented examples
- **AND** the template SHALL follow the format specified in the architecture docs

### Requirement: Configuration versioning
Configuration files SHALL be versioned to handle schema changes.

#### Scenario: Version detection
- **WHEN** loading a configuration file
- **THEN** the system SHALL check the `version` field
- **AND** apply migrations if needed to match current schema

#### Scenario: Backward compatibility
- **WHEN** encountering an older configuration version
- **THEN** the system SHALL attempt to migrate to the current version
- **AND** warn about deprecated fields