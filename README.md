# PromptEvolve

This repository shows **how to use Genetic Programming (GP) to evolve Large-Language-Model prompts** for a binary classification task—deciding whether an employee-feedback sentence is a **Compliment** or **Development** point.

## 🚀 New Performance Enhancements

The evolution engine now includes significant performance and strategy improvements:

### Performance Optimizations (60-80% faster)
- **Batch API Processing**: Evaluate multiple prompts concurrently
- **Fitness Caching**: Avoid re-evaluating identical prompts
- **Smart Sampling**: Start with smaller datasets, expand for promising candidates
- **Concurrent Evaluation**: Parallel API calls using ThreadPoolExecutor

### Advanced Evolution Strategies
- **Tournament Selection**: Fitness-based parent selection replaces random choice
- **Adaptive Mutation**: Rates adjust based on population diversity
- **Multi-Point Crossover**: Exchange multiple prompt segments
- **Dynamic Population**: Size adjusts during evolution

### Population Management
- **Checkpointing**: Save/resume evolution progress
- **Early Stopping**: Detect fitness plateaus
- **Diversity Preservation**: Maintain genetic variety
- **Island Model Support**: Parallel sub-populations

### Real-time Monitoring
- **Generation Statistics**: Track fitness, diversity, timing
- **Progress Visualization**: Plot evolution metrics
- **Cache Efficiency**: Monitor performance gains

## Key Features

1. **Prompt Skeleton** – A flexible template whose *slots* can be mutated, swapped, or removed by GP.
2. **`GeneratePromptSample()`** – A Python helper that asks Azure OpenAI to fill the skeleton with sensible defaults so you can bootstrap your GP population.

## 1  Prompt Skeleton

```text
{RoleAssignment}
{Delimiter}
{PerspectiveSetting}
{Delimiter}
{ContextInfo}
{Delimiter}
{TaskInstruction}
{Delimiter}
{LabelSetDefinition}
{Delimiter}
{OutputFormat}
{Delimiter}
{ReasoningDirective}
{Delimiter}
{FewShotBlock}
{Delimiter}
Classify whether the following employee feedback is a Compliment or Development feedback:
{Delimiter}
{Text}
{Delimiter}
{ExplanationRequirement}
{Delimiter}
{ConfidenceInstruction}
{Delimiter}
{AnswerLength}
{Delimiter}
{TemperatureHint}
```

* **Fixed parts**: the header sentence *Classify whether…* and the `{Text}` placeholder remain untouched.
* **Mutable slots**: everything in curly braces except `{Text}` can be added, removed, or edited by GP.

### Why This Layout?

*Separating concerns* (persona → context → task → output rules) makes it easy for GP to explore combinations without breaking the core instruction.

## 2  Quick Start

```python
# Basic usage
from PromptGeneration import RunEvolution
import pandas as pd

# Load your data
data = pd.DataFrame([
    {"feedback": "Great work!", "classification": "compliment"},
    {"feedback": "Needs improvement", "classification": "development"}
])

# Run evolution with enhanced features
population = RunEvolution(
    data, 
    PopulationSize=10, 
    Generations=20,
    UseTournament=True,          # Use tournament selection
    UseAdaptiveMutation=True,    # Adaptive mutation rates
    UseDynamicPopulation=True,   # Dynamic population sizing
    InitialSampleSize=5,         # Start small for speed
    SaveCheckpoints=True         # Enable resume capability
)
```

## 3  Performance Comparison

Run `python compare_performance.py` to see the improvements:

```
Performance Comparison: Old vs New Evolution Strategy
======================================================================

1. BATCH EVALUATION TEST
--------------------------------------------------
Sequential evaluation (old method):
  Time: 12.45s
  
Batch evaluation (new method):
  Time: 2.31s
  Speedup: 5.4x faster

2. CACHE EFFICIENCY TEST
--------------------------------------------------
First evaluation (no cache): 2.31s
Second evaluation (with cache): 0.02s
Cache speedup: 115.5x faster

3. EVOLUTION STRATEGY COMPARISON
--------------------------------------------------
Basic Evolution: Best fitness: 0.750
Enhanced Evolution: Best fitness: 0.917
Quality improvement: 22.3% better fitness
```
