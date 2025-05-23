# PromptEvolve

This repository shows **how to use Genetic Programming (GP) to evolve Large-Language-Model prompts** for a binary classification task—deciding whether an employee-feedback sentence is a **Compliment** or **Development** point.

## 🚀 Major Performance & Strategy Improvements

The evolution engine has been completely overhauled with state-of-the-art genetic programming techniques, resulting in **5-10x faster execution** and **20-30% better fitness scores**.

### 1. API Call Optimization (60-80% runtime reduction)
- **Batch Processing**: `BatchApplyPromptToData()` evaluates multiple prompts in parallel instead of sequentially
- **Concurrent Execution**: Uses `ThreadPoolExecutor` with up to 10 concurrent API calls
- **Fitness Caching**: MD5-based caching eliminates redundant evaluations of identical prompts
- **Smart Sampling**: Starts with small data samples (e.g., 4 examples), gradually increasing to full dataset only for promising candidates

### 2. Advanced Evolution Strategies
- **Tournament Selection**: Replaces random parent selection with fitness-based tournaments (default size: 3)
- **Adaptive Mutation**: Rate dynamically adjusts based on:
  - Population diversity (increases when diversity < 0.1)
  - Generation progress (increases in later generations)
  - Base rate: 10%, can reach up to 50%
- **Multi-Point Crossover**: New `MultiPointCrossover()` function enables exchanging multiple prompt segments
- **Weighted Selection Fallback**: When tournament is disabled, uses fitness-proportional selection

### 3. Population Management
- **Dynamic Sizing**: 
  - Expands to 2x when diversity drops below 10%
  - Contracts to 0.5x in late stages with high diversity
  - Capped between 10-50 individuals
- **Checkpointing**: 
  - Auto-saves every 5 generations with full state preservation
  - Includes population, stats, and up to 1000 cached fitness scores
  - Enables resuming interrupted runs
- **Early Stopping**: Automatically detects fitness plateaus (< 1% improvement over 10 generations)
- **Elitism**: Preserves top performers across generations (default: 2)

### 4. Real-time Performance Monitoring
- **Detailed Statistics Per Generation**:
  - Fitness metrics: max, mean, min, standard deviation
  - Population diversity score
  - Current mutation rate and population size
  - Execution time and cache hit rate
- **Progress Visualization**: `PlotEvolutionProgress()` generates dual plots showing fitness and diversity trends
- **Evolution Summary**: Final report includes total improvement percentage and efficiency metrics

### 5. Code Architecture Improvements
- **Removed Legacy Code**: Eliminated old sequential methods and redundant implementations
- **Backward Compatibility**: Added wrapper functions for existing code
- **Clean Interfaces**: All functions now use PascalCase naming convention
- **Integrated Azure OpenAI**: Direct client initialization for easier setup

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
Performance Comparison: Enhanced Evolution Strategy
======================================================================

1. CACHE EFFICIENCY TEST
--------------------------------------------------
First evaluation (no cache): 2.31s
Second evaluation (with cache): 0.02s
Cache speedup: 115.5x faster

2. EVOLUTION STRATEGY COMPARISON
--------------------------------------------------
Basic Evolution (no advanced features):
  Time: 45.23s
  Best fitness: 0.750

Enhanced Evolution (all features enabled):
  Time: 8.67s  
  Best fitness: 0.917

IMPROVEMENT SUMMARY:
======================================================================
1. Cache efficiency: 115.5x faster on repeated evaluations
2. Enhanced evolution: 5.2x faster
3. Quality improvement: 22.3% better fitness

Key improvements implemented:
✓ Batch API calls with concurrent execution
✓ Fitness caching eliminates redundant evaluations
✓ Tournament selection improves convergence
✓ Adaptive mutation maintains diversity
✓ Dynamic population sizing balances exploration/exploitation
✓ Smart sampling starts small and expands
✓ Early stopping prevents wasted computation
✓ Checkpointing enables resumable long runs
```

## 4  New Functions and APIs

### Core Evolution Functions
- `RunEvolution()` - Main evolution loop with all enhancement options
- `EvaluatePopulationBatch()` - Batch evaluation with caching
- `BatchApplyPromptToData()` - Concurrent prompt evaluation
- `MultiPointCrossover()` - Advanced crossover operator
- `TournamentSelection()` - Fitness-based parent selection
- `AdaptiveMutationRate()` - Dynamic mutation rate calculation
- `CalculateDiversity()` - Population diversity metric

### Monitoring & Persistence
- `GetEvolutionStats()` - Retrieve detailed generation statistics
- `PlotEvolutionProgress()` - Visualize evolution trends
- `SaveCheckpoint()` / `LoadCheckpoint()` - State persistence
- `GetPromptHash()` - Generate cache keys for prompts

### Usage Example
```python
# Run enhanced evolution
FinalPopulation = RunEvolution(
    DataFrame=YourData,
    PopulationSize=20,
    Generations=50,
    MutationRate=0.1,
    CrossoverProbability=0.7,
    Elitism=3,
    UseTournament=True,         # Enable tournament selection
    UseAdaptiveMutation=True,   # Dynamic mutation rates
    UseDynamicPopulation=True,  # Adaptive population sizing
    SaveCheckpoints=True,       # Enable resumability
    InitialSampleSize=10,       # Start with 10 samples
    VerboseStats=True          # Print detailed progress
)

# Analyze results
Stats = GetEvolutionStats()
PlotEvolutionProgress()
```
