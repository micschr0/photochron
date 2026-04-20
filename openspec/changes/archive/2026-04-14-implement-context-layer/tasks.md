## 1. Dependencies and Setup

- [x] 1.1 Add `ollama` Python dependency to pyproject.toml
- [x] 1.2 Create Ollama client wrapper module with connection handling
- [x] 1.3 Add context-specific configuration to config.yaml
- [x] 1.4 Create test utilities for mocking Ollama responses

## 2. Core LLM Integration

- [x] 2.1 Implement Ollama client with health check and model validation
- [x] 2.2 Create structured prompt templates for decade/season/event/medium analysis
- [x] 2.3 Implement image encoding for Ollama API (base64 or file path)
- [x] 2.4 Create JSON schema validation for LLM outputs
- [x] 2.5 Implement retry logic with exponential backoff for timeouts

## 3. Context Analysis Engine

- [x] 3.1 Create ContextAnalyzer class with main analysis pipeline
- [x] 3.2 Implement decade estimation with confidence scoring
- [x] 3.3 Implement season detection with confidence scoring
- [x] 3.4 Implement event hint detection with confidence scoring
- [x] 3.5 Implement photo medium identification with confidence scoring
- [x] 3.6 Create fallback analysis for failed or low-confidence results

## 4. Database Integration

- [x] 4.1 Implement context table queries for finding photos without context
- [x] 4.2 Create context record insertion/update with transaction support
- [x] 4.3 Implement batch processing with configurable batch size
- [x] 4.4 Add progress tracking for context analysis stage
- [x] 4.5 Create database migration tests for context table schema

## 5. ContextLayerStage Implementation

- [ ] 5.1 Update ContextLayerStage class with proper run() implementation
- [ ] 5.2 Integrate ContextAnalyzer into the stage
- [ ] 5.3 Implement dependency checking (requires face_layer completion)
- [ ] 5.4 Add progress reporting to pipeline framework
- [ ] 5.5 Implement configuration loading for context-specific settings

## 6. Error Handling and Robustness

- [ ] 6.1 Implement comprehensive error handling for Ollama failures
- [ ] 6.2 Create fallback strategies for JSON parsing failures
- [ ] 6.3 Implement model fallback (llava-next:7b → moondream2)
- [ ] 6.4 Add timeout handling for slow LLM responses
- [ ] 6.5 Implement graceful degradation for large photo collections

## 7. Testing

- [ ] 7.1 Create unit tests for Ollama client wrapper
- [ ] 7.2 Create unit tests for ContextAnalyzer with mocked LLM responses
- [ ] 7.3 Create integration tests with sample photos
- [ ] 7.4 Test error handling and retry logic
- [ ] 7.5 Test confidence score propagation and validation
- [ ] 7.6 Test database integration and transaction integrity

## 8. Documentation and Configuration

- [ ] 8.1 Update CLAUDE.md with context layer usage instructions
- [ ] 8.2 Add context layer documentation to agent_docs/
- [ ] 8.3 Document Ollama setup and model installation requirements
- [ ] 8.4 Create example configuration for context analysis
- [ ] 8.5 Add context layer to pipeline documentation