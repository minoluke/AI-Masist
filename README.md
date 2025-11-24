# unitTest - AI Research Experiment Pipeline

A simplified, standalone implementation of AI-Scientist's core draft processing pipeline designed for testing and educational purposes.

## Overview

unitTest is a modular Python package that automates the process of generating, executing, and evaluating experimental code using Large Language Models (LLMs). It provides a 6-phase pipeline for transforming research ideas into executable experiments with automated analysis.

## Features

- **6-Phase Processing Pipeline**: Structured workflow from code generation to analysis
- **Multi-LLM Support**: Compatible with Claude (Anthropic), GPT-4o (OpenAI), and other models
- **Automated Code Execution**: Safe, isolated execution environment with timeout handling
- **Metrics Extraction**: Structured parsing of experimental results
- **Visualization**: Automated plot generation and analysis
- **Vision LLM Integration**: Analyze plots using multimodal AI models

## Architecture

```
Research Idea
    ↓
Phase 1: Code Generation      (LLM generates Python code)
    ↓
Phase 2: Execution            (Interpreter runs code, captures output)
    ↓
Phase 3: Evaluation           (LLM evaluates for bugs)
    ↓
Phase 4: Metrics Extraction   (LLM parses metrics from output)
    ↓
Phase 5: Plot Generation      (Generates visualizations)
    ↓
Phase 6: VLM Analysis         (Vision LLM analyzes plots)
    ↓
Results Node
```

## Directory Structure

```
unitTest/
├── core/                     # Core data structures
│   ├── node.py              # Experiment node representation
│   ├── metric.py            # Metrics and evaluation
│   └── execution_result.py  # Code execution results
├── llm/                      # LLM integration
│   ├── backend/             # API backends (Anthropic, OpenAI)
│   └── function_specs.py    # LLM function calling specs
├── executor/
│   └── interpreter.py       # Code execution engine
├── phases/                   # 6-phase pipeline
│   ├── code_generator.py    # Phase 1: Generate code
│   ├── result_evaluator.py  # Phase 3: Evaluate results
│   ├── metrics_extractor.py # Phase 4: Extract metrics
│   ├── plot_generator.py    # Phase 5: Generate plots
│   └── vlm_analyzer.py      # Phase 6: Analyze plots
├── utils/                    # Utilities
│   └── response.py          # Response parsing
├── draft_processor.py       # Main orchestrator
└── example_usage.py         # Usage example
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/unitTest.git
cd unitTest

# Install dependencies
pip install anthropic openai matplotlib seaborn
```

## Usage

### Basic Example

```python
from unitTest import DraftProcessor

# Configure the processor
config = {
    "backend": "anthropic",  # or "openai"
    "model_code_gen": "claude-3-5-sonnet-20241022",
    "model_eval": "gpt-4o-2024-11-20",
    "timeout": 3600,
    "temperature": 1.0
}

# Initialize processor
processor = DraftProcessor(
    task_desc="Implement a simple MNIST classifier using PyTorch",
    evaluation_metrics=["accuracy", "loss"],
    cfg=config
)

# Process the draft
node = processor.process_draft(working_dir="./working")

# Access results
print(f"Metrics: {node.metrics}")
print(f"VLM Feedback: {node.vlm_feedback}")
```

### See `example_usage.py` for more examples

## Configuration

### Required Environment Variables

```bash
# For Anthropic (Claude)
export ANTHROPIC_API_KEY="your-api-key"

# For OpenAI (GPT-4o)
export OPENAI_API_KEY="your-api-key"
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `backend` | LLM backend ("anthropic" or "openai") | "anthropic" |
| `model_code_gen` | Model for code generation | "claude-3-5-sonnet-20241022" |
| `model_eval` | Model for evaluation | "gpt-4o-2024-11-20" |
| `timeout` | Code execution timeout (seconds) | 3600 |
| `temperature` | LLM temperature | 1.0 |

## Core Components

### Node
Represents a single experiment attempt with:
- Generated code and execution plan
- Execution results (stdout, stderr, exceptions)
- Extracted metrics
- Generated plots
- VLM analysis feedback

### ExecutionResult
Captures comprehensive execution information:
- Standard output and error streams
- Exception details
- Execution timing
- Success/failure status

### MetricValue
Structured metric storage with:
- Metric name and value
- Confidence score
- Ordering information
- Type validation

## Requirements

- Python 3.11+
- anthropic
- openai
- matplotlib
- seaborn

## License

MIT License

## Acknowledgments

This project is inspired by and derived from the AI-Scientist framework for autonomous scientific research.
