## ADDED Requirements

### Requirement: Analyze photo context using vision LLM
The system SHALL analyze downsampled photos using a vision LLM to extract visual context information for chronological dating. The analysis MUST include decade estimate, season, event hints, and photo medium identification. Each analysis result MUST include confidence scores between 0.0 and 1.0.

#### Scenario: Single photo context analysis
- **WHEN** a downsampled photo is analyzed by the context layer
- **THEN** the system returns structured JSON with decade estimate, season, event hints, photo medium, and confidence scores

#### Scenario: No visual context detected
- **WHEN** a photo contains no discernible visual context (e.g., abstract image)
- **THEN** the system returns minimal context data with low confidence scores and appropriate null values

#### Scenario: Multiple context elements detected
- **WHEN** a photo contains multiple context elements (e.g., summer wedding with 1980s fashion)
- **THEN** the system returns all detected elements with appropriate confidence scores

### Requirement: Extract decade estimate with confidence
The system SHALL estimate the decade when a photo was taken based on visual cues (fashion, technology, architecture, etc.). The estimate MUST be expressed as a range (e.g., "1985-1990") and MUST include a confidence score between 0.0 and 1.0.

#### Scenario: Clear decade cues detected
- **WHEN** a photo contains clear decade indicators (e.g., 1990s fashion, 1970s car models)
- **THEN** the system returns a decade estimate with confidence > 0.7

#### Scenario: Ambiguous decade cues
- **WHEN** a photo contains ambiguous or conflicting decade indicators
- **THEN** the system returns a decade estimate with confidence < 0.5 and flags for review

#### Scenario: Modern photo detection
- **WHEN** a photo contains modern elements (smartphones, contemporary fashion)
- **THEN** the system returns a recent decade estimate (e.g., "2015-2020") with appropriate confidence

### Requirement: Identify season from visual cues
The system SHALL identify the season (spring, summer, autumn, winter) depicted in a photo based on visual indicators (foliage, weather, clothing, etc.). The season identification MUST include a confidence score between 0.0 and 1.0.

#### Scenario: Clear seasonal indicators
- **WHEN** a photo shows clear seasonal indicators (snow, autumn leaves, summer beach)
- **THEN** the system returns the correct season with confidence > 0.7

#### Scenario: Indoor or seasonless photo
- **WHEN** a photo shows indoor scenes or lacks seasonal indicators
- **THEN** the system returns null for season with confidence 0.0

#### Scenario: Transitional season detection
- **WHEN** a photo shows transitional season indicators (early spring, late autumn)
- **THEN** the system returns the appropriate season with moderate confidence (0.4-0.6)

### Requirement: Detect event hints
The system SHALL detect potential events depicted in photos (wedding, birthday, graduation, holiday, etc.). Event hints MUST be returned as text descriptions and MUST include confidence scores between 0.0 and 1.0.

#### Scenario: Clear event indicators
- **WHEN** a photo shows clear event indicators (wedding dress, birthday cake, graduation gown)
- **THEN** the system returns the event hint with confidence > 0.6

#### Scenario: No event detected
- **WHEN** a photo shows no clear event indicators
- **THEN** the system returns null for event hint with confidence 0.0

#### Scenario: Multiple possible events
- **WHEN** a photo could depict multiple events (e.g., formal gathering could be wedding or graduation)
- **THEN** the system returns the most likely event with appropriate confidence or returns null if uncertain

### Requirement: Identify photo medium
The system SHALL identify the photo medium type (print_scan, digital, polaroid, film_negative, etc.) based on visual characteristics. The medium identification MUST include a confidence score between 0.0 and 1.0.

#### Scenario: Digital photo characteristics
- **WHEN** a photo shows digital photo characteristics (clean edges, no film grain, modern aspect ratio)
- **THEN** the system identifies it as "digital" with confidence > 0.7

#### Scenario: Print scan characteristics
- **WHEN** a photo shows print scan characteristics (film grain, border artifacts, color fading)
- **THEN** the system identifies it as "print_scan" with confidence > 0.7

#### Scenario: Polaroid characteristics
- **WHEN** a photo shows Polaroid characteristics (white border, square format, distinctive color palette)
- **THEN** the system identifies it as "polaroid" with confidence > 0.7

### Requirement: Use structured JSON output from LLM
The system SHALL use structured JSON prompting to ensure consistent output format from the vision LLM. The JSON schema MUST match the expected database schema and MUST be validated before storage.

#### Scenario: Valid JSON output
- **WHEN** the LLM returns valid JSON matching the expected schema
- **THEN** the system parses and stores the data successfully

#### Scenario: Invalid JSON output
- **WHEN** the LLM returns invalid or malformed JSON
- **THEN** the system retries with a simplified prompt and logs the error

#### Scenario: JSON schema mismatch
- **WHEN** the LLM returns JSON with missing or incorrect fields
- **THEN** the system applies default values for missing fields and logs schema violations

### Requirement: Implement retry logic for LLM failures
The system SHALL implement retry logic for LLM inference failures, JSON parsing errors, and timeouts. The retry logic MUST include fallback strategies and MUST prevent infinite retry loops.

#### Scenario: LLM timeout on first attempt
- **WHEN** the LLM times out on the first analysis attempt
- **THEN** the system retries once with the same parameters

#### Scenario: JSON parsing failure after retry
- **WHEN** the LLM returns unparseable JSON after retry
- **THEN** the system falls back to a simpler model or stores minimal data with low confidence

#### Scenario: Successful retry
- **WHEN** the first LLM attempt fails but retry succeeds
- **THEN** the system stores the successful result and logs the initial failure

### Requirement: Store context data in feature store
The system SHALL store all context analysis results in the `context` table of the SQLite feature store. Each record MUST include photo_id, decade, decade_confidence, season, event_hint, photo_medium, raw_json, and timestamps.

#### Scenario: New context record creation
- **WHEN** a photo is successfully analyzed for the first time
- **THEN** a new record is inserted into the `context` table with all extracted data

#### Scenario: Context record update
- **WHEN** a photo is re-analyzed with updated context information
- **THEN** the existing `context` record is updated with new data

#### Scenario: Database transaction integrity
- **WHEN** context analysis fails midway through a batch of photos
- **THEN** no partial context data is committed to the database

### Requirement: Integrate with pipeline framework
The context layer SHALL implement the `PipelineStage` abstract base class and SHALL declare `face_layer` as a dependency. The stage SHALL report progress through the pipeline's progress tracking system.

#### Scenario: Stage runs after face layer
- **WHEN** the pipeline executes with both face layer and context layer enabled
- **THEN** the context layer runs only after the face layer completes successfully

#### Scenario: Progress reporting during processing
- **WHEN** processing a batch of photos
- **THEN** the stage reports incremental progress (e.g., "5/100 photos analyzed")

#### Scenario: Configuration validation
- **WHEN** the context layer is initialized
- **THEN** it validates Ollama configuration and model availability before processing