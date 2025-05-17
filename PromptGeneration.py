import random
import re
import pandas as pd

def GeneratePromptSample(RoleAssignment: str = "You are an expert in employee-feedback analysis.") -> dict:
    """
    Calls Azure OpenAI once and returns a dictionary describing a prompt.
    The returned mapping includes every slot from the skeleton, and the
    ``Text`` field is preserved as a placeholder.

    Parameters
    ----------
    RoleAssignment : str, optional
        A custom persona you want at the top of the prompt.

    Returns
    -------
    dict
        A mapping of prompt slots to the generated text. The placeholder
        ``Text`` remains in the output so you can insert your own content
        later.
    """


    PromptSkeleton = """{RoleAssignment}
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
{TemperatureHint}"""

    SystemMessage = {
        "role": "system",
        "content": "You are a top-tier prompt-engineering assistant."
    }

    UserMessage = {
        "role": "user",
        "content": (
            "Using the skeleton below, generate a COMPLETE prompt for classifying "
            "talent feedback into two labels: Compliment and Development. "
            "• Replace every slot **except {Text} and the {Delimiter} tokens** with suitable content. "
            "• Keep placeholders wrapped in curly braces exactly as shown. "
            "• Output ONLY the finished prompt, nothing else.\n\n"
            + PromptSkeleton.replace("{RoleAssignment}", RoleAssignment)
        )
    }

    Response = client.chat.completions.create(
        model = 'gpt-4o',
        messages = [SystemMessage, UserMessage],
        temperature = 1,
    )

    PromptResult = Response.choices[0].message.content.strip()

    return ParsePromptWithDelimiter(PromptResult)


def ParsePromptWithDelimiter(PromptText: str) -> dict:
    """Parse a prompt string that uses a consistent delimiter between slots."""

    PromptLines = [Line.rstrip() for Line in PromptText.splitlines()]
    if len(PromptLines) < 3:
        return {}

    DelimiterToken = PromptLines[1]

    Segments = []
    CurrentSegment = [PromptLines[0]]
    for Line in PromptLines[1:]:
        if Line.strip() == DelimiterToken:
            Segments.append("\n".join(CurrentSegment).strip())
            CurrentSegment = []
        else:
            CurrentSegment.append(Line)
    if CurrentSegment:
        Segments.append("\n".join(CurrentSegment).strip())

    if len(Segments) < 14:
        return {}

    ParsedPrompt = {
        "RoleAssignment": Segments[0],
        "PerspectiveSetting": Segments[1],
        "ContextInfo": Segments[2],
        "TaskInstruction": Segments[3],
        "LabelSetDefinition": Segments[4],
        "OutputFormat": Segments[5],
        "ReasoningDirective": Segments[6],
        "FewShotBlock": Segments[7],
        "Text": Segments[9],
        "ExplanationRequirement": Segments[10],
        "ConfidenceInstruction": Segments[11],
        "AnswerLength": Segments[12],
        "TemperatureHint": Segments[13],
    }

    return ParsedPrompt


PromptSkeletonKeys = [
    "RoleAssignment",
    "PerspectiveSetting",
    "ContextInfo",
    "TaskInstruction",
    "LabelSetDefinition",
    "OutputFormat",
    "ReasoningDirective",
    "FewShotBlock",
    "ExplanationRequirement",
    "ConfidenceInstruction",
    "AnswerLength",
    "TemperatureHint",
]


def Crossover(ParentOne: dict, ParentTwo: dict, Probability: float = 0.5) -> dict:
    """Perform uniform crossover on prompt dictionaries.

    Each slot defined in :data:`PromptSkeletonKeys` is randomly selected from
    either ``ParentOne`` or ``ParentTwo`` based on ``Probability``. The ``Text``
    element is copied from ``ParentOne`` without modification.
    """

    ChildPrompt = {}
    for SlotName in PromptSkeletonKeys:
        if random.random() < Probability:
            ChildPrompt[SlotName] = ParentOne.get(SlotName, "")
        else:
            ChildPrompt[SlotName] = ParentTwo.get(SlotName, "")

    ChildPrompt["Text"] = ParentOne.get("Text", ParentTwo.get("Text", ""))

    return ChildPrompt

def MutatePromptField(Prompt: dict, Field: str) -> None:
    """Rephrase a slot in ``Prompt`` using Azure OpenAI."""

    if Field not in Prompt or not Prompt[Field]:
        return

    SystemMessage = {
        "role": "system",
        "content": "You are a helpful rewriting assistant.",
    }

    UserMessage = {
        "role": "user",
        "content": f"Rephrase the following so it keeps the same meaning:\n{Prompt[Field]}",
    }

    Response = client.chat.completions.create(
        model = 'gpt-4o',
        messages = [SystemMessage, UserMessage],
        temperature = 1,
    )

    Prompt[Field] = Response.choices[0].message.content.strip()

    return Prompt

def CombineString(dictPrompt):

  CombinedString = ""
  for value in dictPrompt.values():
      if isinstance(value, str):
          CombinedString += value + "\n"

  return CombinedString


def ApplyPromptToData(DataFrame: pd.DataFrame, Prompt: str) -> pd.DataFrame:
    """Apply a classification ``Prompt`` to each feedback entry in ``DataFrame``.

    Parameters
    ----------
    DataFrame : pandas.DataFrame
        Contains ``feedback`` and ``classification`` columns.
    Prompt : str
        The prompt text that includes a ``{Text}`` placeholder.

    Returns
    -------
    pandas.DataFrame
        A copy of ``DataFrame`` with a new ``Prediction`` column.
    """

    Predictions = []
    for Feedback in DataFrame["feedback"]:
        FilledPrompt = Prompt.replace("{Text}", Feedback)

        SystemMessage = {
            "role": "system",
            "content": "You are a helpful classification assistant.",
        }

        UserMessage = {"role": "user", "content": FilledPrompt}

        Response = client.chat.completions.create(
            model='gpt-4o',
            messages=[SystemMessage, UserMessage],
            temperature=0,
        )

        Prediction = Response.choices[0].message.content.strip()
        Predictions.append(Prediction)

    Result = DataFrame.copy()
    Result["Prediction"] = Predictions
    return Result


def CalculateFitnessScore(ResultFrame: pd.DataFrame) -> float:
    """Return classification accuracy from a result DataFrame.

    The ``Prediction`` column can contain full sentences. This function extracts
    either ``compliment`` or ``development`` from each prediction and compares it
    to the ``classification`` column, ignoring case.
    """

    Pattern = re.compile(r"(compliment|development)", re.IGNORECASE)
    CorrectCount = 0

    for Prediction, Actual in zip(ResultFrame.get("Prediction", []), ResultFrame.get("classification", [])):
        if not isinstance(Prediction, str):
            continue
        Match = Pattern.search(Prediction)
        if not Match:
            continue

        NormalizedPred = Match.group(1).lower()
        if isinstance(Actual, str) and NormalizedPred == Actual.strip().lower():
            CorrectCount += 1

    FitnessScore = CorrectCount / len(ResultFrame) if len(ResultFrame) else 0.0
    return FitnessScore


def GeneratePopulation(PopulationSize: int) -> list:
    """Create an initial population of prompts."""

    return [GeneratePromptSample() for _ in range(PopulationSize)]


def MutatePrompt(Prompt: dict, MutationRate: float) -> dict:
    """Mutate fields within ``Prompt`` with probability ``MutationRate``."""

    for Field in PromptSkeletonKeys:
        if random.random() < MutationRate:
            MutatePromptField(Prompt, Field)
    return Prompt


def EvaluatePrompt(Prompt: dict, DataFrame: pd.DataFrame) -> float:
    """Compute the fitness score for ``Prompt`` on ``DataFrame``."""

    PromptText = CombineString(Prompt)
    ResultFrame = ApplyPromptToData(DataFrame, PromptText)
    return CalculateFitnessScore(ResultFrame)


def RunEvolution(DataFrame: pd.DataFrame, PopulationSize: int, Generations: int,
                 MutationRate: float = 0.1, CrossoverProbability: float = 0.5,
                 Elitism: int = 1) -> list:
    """Evolve prompts over multiple generations."""

    Population = GeneratePopulation(PopulationSize)

    for _ in range(Generations):
        Scores = [EvaluatePrompt(Prompt, DataFrame) for Prompt in Population]
        ScoredPopulation = list(zip(Population, Scores))
        ScoredPopulation.sort(key=lambda Item: Item[1], reverse=True)

        NewPopulation = [Item[0] for Item in ScoredPopulation[:Elitism]]

        while len(NewPopulation) < PopulationSize:
            ParentOne = random.choice(Population)
            ParentTwo = random.choice(Population)
            Child = Crossover(ParentOne, ParentTwo, CrossoverProbability)
            Child = MutatePrompt(Child, MutationRate)
            NewPopulation.append(Child)

        Population = NewPopulation

    return Population
