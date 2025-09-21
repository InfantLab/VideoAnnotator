# VideoAnnotator Testing Standards

This document outlines the testing standards and practices for the VideoAnnotator project.

## Testing Framework

The VideoAnnotator project uses **pytest** as its primary testing framework. All tests should be written using pytest conventions and features.

## Test Organization

### Directory Structure

```
tests/
├── conftest.py                        # Pytest configuration and fixtures
├── test_all_pipelines.py              # Test runner and organization
├── test_face_pipeline.py              # Face Analysis Pipeline tests
├── test_audio_pipeline.py             # Audio Processing Pipeline tests
├── test_scene_pipeline.py             # Scene Detection Pipeline tests
├── test_person_pipeline.py            # Person Tracking Pipeline tests
├── test_audio_speech_pipeline.py      # Diarization and Speech Recognition tests
├── test_pipelines.py                  # Legacy compatibility wrapper (deprecated)
├── test_schemas.py                     # Schema validation tests  
├── test_integration.py                # Integration tests
├── test_performance.py               # Performance benchmarks
```

### Test Categories

Tests are organized using pytest marks for easy categorization:

1. **Unit Tests** (`@pytest.mark.unit`)
   - Test individual pipeline components in isolation
   - Test configuration validation
   - Test error handling
   - Mock external dependencies
   - Fast execution (< 1 second per test)

2. **Integration Tests** (`@pytest.mark.integration`)
   - Test component integration with real dependencies
   - Test end-to-end pipeline workflows
   - Test file I/O operations
   - May require external models/services

3. **Performance Tests** (`@pytest.mark.performance`)
   - Benchmark pipeline performance
   - Memory usage testing
   - Scalability testing
   - Resource consumption validation

4. **Schema Tests** (in dedicated files)
   - Test Pydantic model validation
   - Test data serialization/deserialization
   - Test schema migrations

### Pipeline-Specific Test Files

The VideoAnnotator test suite uses a modular approach with dedicated test files for each pipeline:

#### Individual Pipeline Tests
- **Face Analysis** (`test_face_pipeline.py`): Tests face detection, emotion analysis, JSON serialization fixes
- **Audio Processing** (`test_audio_pipeline.py`): Tests audio feature extraction, schema validation fixes
- **Scene Detection** (`test_scene_pipeline.py`): Tests scene detection and classification
- **Person Tracking** (`test_person_pipeline.py`): Tests YOLO integration, tracking, pose estimation
- **Audio Speech** (`test_audio_speech_pipeline.py`): Tests diarization and speech recognition

#### Test Organization Benefits
- **Focused Testing**: Each file contains tests for a single pipeline
- **Easier Maintenance**: Smaller, focused test files are easier to maintain
- **Parallel Execution**: Pipeline tests can be run independently
- **Clear Responsibilities**: Each test file has a clear scope and purpose

#### Running Modular Tests
```bash
# Run tests for a specific pipeline
python -m pytest tests/test_face_pipeline.py -v

# Run specific test categories across all pipelines  
python -m pytest tests/ -m unit -v
python -m pytest tests/ -m integration -v
python -m pytest tests/ -m performance -v

# Run all pipeline tests through the test runner
python -m pytest tests/test_all_pipelines.py -v
```

## Test File Standards

### Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*` (e.g., `TestScenePipeline`)
- Test methods: `test_*` (e.g., `test_scene_detection_threshold`)

### Test Structure

```python
class TestPipelineComponent:
    """Test cases for pipeline component."""
    
    def test_initialization(self):
        """Test component initialization."""
        # Arrange
        config = ComponentConfig(param=value)
        
        # Act
        component = Component(config)
        
        # Assert
        assert component.config.param == value
    
    def test_process_valid_input(self):
        """Test processing with valid input."""
        # Test implementation
        pass
    
    def test_process_invalid_input(self):
        """Test error handling with invalid input."""
        # Test implementation
        pass
```

## Fixtures and Mocking

### Common Fixtures (`conftest.py`)

- `sample_video`: Creates temporary test video files
- `sample_audio`: Creates temporary test audio files
- `temp_output_dir`: Creates temporary output directories
- `mock_models`: Mocks ML model dependencies

### Mocking Guidelines

1. **Mock External Dependencies**: Always mock external APIs, file systems, and ML models
2. **Use pytest-mock**: Prefer `pytest-mock` plugin for mocking
3. **Mock at Module Level**: Mock imports at the module level when possible

```python
@patch('src.pipelines.scene_detection.cv2')
def test_video_processing(mock_cv2, sample_video):
    """Test video processing with mocked OpenCV."""
    # Test implementation
```

## Test Data Management

### Test Data Location

Store test data in `tests/data/` directory:

```
tests/data/
├── videos/
│   ├── sample_short.mp4    # 5-second test video
│   └── sample_long.mp4     # 30-second test video
├── audio/
│   ├── sample_speech.wav   # Speech audio sample
│   └── sample_music.wav    # Music audio sample
└── expected_outputs/
    ├── scene_results.json  # Expected scene detection results
    └── person_results.json # Expected person tracking results
```

### Data Size Limits

- Keep test videos under 10MB
- Keep test audio files under 5MB
- Use compressed formats (H.264, MP3)

## Coverage Requirements

- **Minimum Coverage**: 80% overall
- **Pipeline Coverage**: 90% for core pipeline code
- **Schema Coverage**: 95% for data validation code

### Coverage Commands

```bash
# Run tests with coverage
pytest --cov=src tests/

# Generate HTML coverage report
pytest --cov=src --cov-report=html tests/

# Check coverage requirements
pytest --cov=src --cov-fail-under=80 tests/
```

## Performance Testing

### Benchmark Standards

- Test with standardized input sizes
- Measure processing time per frame
- Monitor memory usage
- Test on different hardware configurations

### Performance Metrics

```python
@pytest.mark.performance
def test_pipeline_performance(benchmark, sample_video):
    """Benchmark pipeline performance."""
    def run_pipeline():
        return pipeline.process_video(sample_video)
    
    result = benchmark(run_pipeline)
    assert result is not None
```

## Continuous Integration

### Test Execution

Tests should pass in the following environments:
- Python 3.8, 3.9, 3.10, 3.11
- Windows, Linux, macOS
- CPU and GPU environments

### CI Configuration

```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: [3.8, 3.9, 3.10, 3.11]
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt
      - name: Run tests
        run: pytest --cov=src tests/
```

## Test Documentation

### Docstring Standards

Every test should have a clear docstring explaining:
- What is being tested
- Expected behavior
- Any special setup requirements

```python
def test_scene_detection_threshold(self):
    """
    Test scene detection with different threshold values.
    
    Verifies that:
    - Higher thresholds result in fewer scene changes
    - Lower thresholds result in more scene changes
    - Invalid thresholds raise appropriate errors
    """
    # Test implementation
```

### Test Comments

- Comment complex test setup
- Explain non-obvious assertions
- Document known limitations or edge cases

## Error Testing

### Exception Testing

```python
def test_invalid_video_path(self):
    """Test handling of invalid video paths."""
    with pytest.raises(FileNotFoundError):
        pipeline.process_video("nonexistent_video.mp4")

def test_invalid_configuration(self):
    """Test configuration validation."""
    with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
        ScenePipelineConfig(threshold=2.0)
```

### Edge Cases

Always test:
- Empty inputs
- Null/None values
- Extremely large inputs
- Malformed data
- Network failures (for external APIs)

## Integration with Main Testing Framework

### Adding New Tests

1. **Identify the appropriate test file** in `tests/`
2. **Add test class** if testing a new component
3. **Add test methods** following naming conventions
4. **Use existing fixtures** from `conftest.py`
5. **Follow documentation standards**

### Example Addition

```python
# In tests/test_pipelines.py
class TestNewPipeline:
    """Test cases for new pipeline component."""
    
    def test_new_pipeline_initialization(self):
        """Test new pipeline initialization."""
        # Test implementation
    
    def test_new_pipeline_processing(self, sample_video):
        """Test new pipeline video processing."""
        # Test implementation
```

## Best Practices

1. **Test First**: Write tests before implementing features
2. **Keep Tests Simple**: Each test should focus on one behavior
3. **Use Descriptive Names**: Test names should clearly describe what's being tested
4. **Mock External Dependencies**: Don't test external libraries
5. **Test Edge Cases**: Include boundary conditions and error cases
6. **Regular Maintenance**: Keep tests up to date with code changes
7. **Performance Awareness**: Monitor test execution time
8. **Documentation**: Keep test documentation current

## Running Tests

### Local Development

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_pipelines.py

# Run specific test class
pytest tests/test_pipelines.py::TestScenePipeline

# Run specific test method
pytest tests/test_pipelines.py::TestScenePipeline::test_initialization

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src

# Run performance tests only
pytest -m performance
```

### Debug Mode

```bash
# Run tests with debug output
pytest -s --log-cli-level=DEBUG

# Run tests with pdb on failure
pytest --pdb

# Run tests with pytest-xvs for detailed output
pytest -xvs
```

This testing framework ensures consistent, reliable, and maintainable tests across the VideoAnnotator project.
