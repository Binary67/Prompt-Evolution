# PromptEvolve

This repository shows **how to use Genetic Programming (GP) to evolve Large-Language-Model prompts** for text classification tasks. Originally designed for binary classification of employee feedback, it now supports **any classification task** with configurable labels, data columns, and task descriptions.

The project contains three key ideas:

1. **Prompt Skeleton** – A flexible template whose *slots* can be mutated, swapped, or removed by GP.
2. **`GeneratePromptSample()`** – A Python helper that asks Azure OpenAI to fill the skeleton with sensible defaults so you can bootstrap your GP population.
3. **`ClassificationTaskConfig`** – A configuration class that makes the system generic for any classification task.

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
{TaskDescription}
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

* **Fixed parts**: the `{TaskDescription}` (configurable per task) and the `{Text}` placeholder remain untouched.
* **Mutable slots**: everything in curly braces except `{Text}` and `{TaskDescription}` can be added, removed, or edited by GP.

### Why This Layout?

*Separating concerns* (persona → context → task → output rules) makes it easy for GP to explore combinations without breaking the core instruction.

## 2  Generic Classification Support

The system now supports any text classification task through the `ClassificationTaskConfig` class:

```python
config = ClassificationTaskConfig(
    Labels=["positive", "negative", "neutral"],  # Your classification labels
    TaskDescription="Classify the sentiment of the following text:",  # Custom task description
    DataColumnName="text",  # Column containing text to classify
    LabelColumnName="label"  # Column containing true labels
)
```

### Supported Use Cases

1. **Employee Feedback Classification** (original use case)
   - Labels: `["compliment", "development"]`
   - Classifies feedback as complimentary or developmental

2. **Sentiment Analysis**
   - Labels: `["positive", "negative", "neutral"]`
   - Analyzes text sentiment

3. **Spam Detection**
   - Labels: `["spam", "ham"]`
   - Identifies spam vs legitimate emails

4. **Any Custom Classification**
   - Define your own labels and task description
   - Works with binary or multi-class classification

### Key Improvements

- **Flexible Label System**: No longer hardcoded to "compliment/development" - use any labels
- **Configurable Data Columns**: Specify your DataFrame column names instead of fixed "feedback" and "classification"
- **Dynamic Pattern Matching**: Automatically creates regex patterns based on your configured labels
- **Task-Specific Descriptions**: Customize the classification instruction for your specific use case

## 3  Usage Example

```python
import pandas as pd
from PromptGeneration import RunEvolution, ClassificationTaskConfig

# Prepare your data
data = pd.DataFrame([
    {"text": "This product is amazing!", "label": "positive"},
    {"text": "Terrible experience", "label": "negative"},
    # ... more examples
])

# Configure your classification task
config = ClassificationTaskConfig(
    Labels=["positive", "negative"],
    TaskDescription="Determine the sentiment of the following review:",
    DataColumnName="text",
    LabelColumnName="label"
)

# Run the evolution
population_size = 10
generations = 5
final_population = RunEvolution(data, config, population_size, generations)
```

The system will evolve prompts optimized for your specific classification task, automatically handling the label extraction and accuracy calculation based on your configuration.
