# PromptEvolve

This repository shows **how to use Genetic Programming (GP) to evolve Large-Language-Model prompts** for a binary classification task—deciding whether an employee-feedback sentence is a **Compliment** or **Development** point.

The project contains two key ideas:

1. **Prompt Skeleton** – A flexible template whose *slots* can be mutated, swapped, or removed by GP.
2. **`GeneratePromptSample()`** – A Python helper that asks Azure OpenAI to fill the skeleton with sensible defaults so you can bootstrap your GP population.

## 1  Prompt Skeleton

```text
{RoleAssignment}
{PerspectiveSetting}
{ContextInfo}
{TaskInstruction}
{LabelSetDefinition}
{OutputFormat}
{ReasoningDirective}
{FewShotBlock}
{Delimiter}
Classify whether the following employee feedback is a Compliment or Development feedback:
{Delimiter}
{Text}
{Delimiter}
{ExplanationRequirement}
{ConfidenceInstruction}
{AnswerLength}
{TemperatureHint}
```

* **Fixed parts**: the header sentence *Classify whether…* and the `{Text}` placeholder remain untouched.
* **Mutable slots**: everything in curly braces except `{Text}` can be added, removed, or edited by GP.

### Why This Layout?

*Separating concerns* (persona → context → task → output rules) makes it easy for GP to explore combinations without breaking the core instruction.
