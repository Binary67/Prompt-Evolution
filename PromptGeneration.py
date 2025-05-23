import random
import re
import pandas as pd
from openai import AzureOpenAI
import os

os.environ["AZURE_OPENAI_API_KEY"] = "3026be1058fa4f0c9e3416d3d8227657"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ptsg-5talendopenai01.openai.azure.com/"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01"
os.environ["AZURE_OPENAI_DEPLOYMENT"] = "myTalentX_GPT4omini"

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

class ClassificationTaskConfig:
    """Configuration for classification tasks."""
    def __init__(self, Labels, TaskDescription=None, DataColumnName="text", LabelColumnName="label"):
        """
        Parameters
        ----------
        Labels : list
            List of classification labels (e.g., ["positive", "negative"] or ["compliment", "development"])
        TaskDescription : str, optional
            Custom task description. If None, generates default based on labels
        DataColumnName : str
            Name of the column containing text to classify (default: "text")
        LabelColumnName : str
            Name of the column containing true labels (default: "label")
        """
        self.Labels = Labels
        self.DataColumnName = DataColumnName
        self.LabelColumnName = LabelColumnName
        
        if TaskDescription is None:
            LabelStr = " or ".join(Labels)
            self.TaskDescription = f"Classify the following text as {LabelStr}:"
        else:
            self.TaskDescription = TaskDescription

def GeneratePromptSample(TaskConfig: ClassificationTaskConfig, RoleAssignment: str = "You are an expert classification assistant.") -> dict:
    """
    Calls Azure OpenAI once and returns a dictionary describing a prompt.
    The returned mapping includes every slot from the skeleton, and the
    ``Text`` field is preserved as a placeholder.

    Parameters
    ----------
    TaskConfig : ClassificationTaskConfig
        Configuration object containing labels and task description
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
{TemperatureHint}"""

    SystemMessage = {
        "role": "system",
        "content": "You are a top-tier prompt-engineering assistant."
    }

    LabelStr = " and ".join(TaskConfig.Labels)
    UserMessage = {
        "role": "user",
        "content": (
            f"Using the skeleton below, generate a COMPLETE prompt for classifying "
            f"text into these labels: {LabelStr}. "
            "• Replace every slot **except {Text}, {TaskDescription}, and the {Delimiter} tokens** with suitable content. "
            "• Keep placeholders wrapped in curly braces exactly as shown. "
            "• Output ONLY the finished prompt, nothing else.\n\n"
            + PromptSkeleton.replace("{RoleAssignment}", RoleAssignment).replace("{TaskDescription}", TaskConfig.TaskDescription)
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


def ApplyPromptToData(DataFrame: pd.DataFrame, Prompt: str, TaskConfig: ClassificationTaskConfig) -> pd.DataFrame:
    """Apply a classification ``Prompt`` to each text entry in ``DataFrame``.

    Parameters
    ----------
    DataFrame : pandas.DataFrame
        Contains data and label columns as specified in TaskConfig.
    Prompt : str
        The prompt text that includes a ``{Text}`` placeholder.
    TaskConfig : ClassificationTaskConfig
        Configuration specifying column names and labels.

    Returns
    -------
    pandas.DataFrame
        A copy of ``DataFrame`` with a new ``Prediction`` column.
    """

    Predictions = []
    for TextData in DataFrame[TaskConfig.DataColumnName]:
        FilledPrompt = Prompt.replace("{Text}", TextData)

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


def CalculateFitnessScore(ResultFrame: pd.DataFrame, TaskConfig: ClassificationTaskConfig) -> float:
    """Return classification accuracy from a result DataFrame.

    The ``Prediction`` column can contain full sentences. This function extracts
    any of the valid labels from each prediction and compares it
    to the label column, ignoring case.
    """

    # Create pattern from valid labels
    LabelPattern = "|".join(re.escape(Label) for Label in TaskConfig.Labels)
    Pattern = re.compile(f"({LabelPattern})", re.IGNORECASE)
    CorrectCount = 0

    for Prediction, Actual in zip(ResultFrame.get("Prediction", []), ResultFrame.get(TaskConfig.LabelColumnName, [])):
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


def GeneratePopulation(PopulationSize: int, TaskConfig: ClassificationTaskConfig) -> list:
    """Create an initial population of prompts."""

    return [GeneratePromptSample(TaskConfig) for _ in range(PopulationSize)]


def MutatePrompt(Prompt: dict, MutationRate: float) -> dict:
    """Mutate fields within ``Prompt`` with probability ``MutationRate``."""

    for Field in PromptSkeletonKeys:
        if random.random() < MutationRate:
            MutatePromptField(Prompt, Field)
    return Prompt


def EvaluatePrompt(Prompt: dict, DataFrame: pd.DataFrame, TaskConfig: ClassificationTaskConfig) -> float:
    """Compute the fitness score for ``Prompt`` on ``DataFrame``."""

    PromptText = CombineString(Prompt)
    ResultFrame = ApplyPromptToData(DataFrame, PromptText, TaskConfig)
    return CalculateFitnessScore(ResultFrame, TaskConfig)


def RunEvolution(DataFrame: pd.DataFrame, TaskConfig: ClassificationTaskConfig, 
                 PopulationSize: int, Generations: int,
                 MutationRate: float = 0.1, CrossoverProbability: float = 0.5,
                 Elitism: int = 1) -> list:
    """Evolve prompts over multiple generations."""

    Population = GeneratePopulation(PopulationSize, TaskConfig)

    for _ in range(Generations):
        Scores = [EvaluatePrompt(Prompt, DataFrame, TaskConfig) for Prompt in Population]
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
