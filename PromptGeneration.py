import random
import re
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import time
import numpy as np
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

# Global cache for fitness scores
FitnessCache = {}
GenerationStats = []

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


def MultiPointCrossover(ParentOne: dict, ParentTwo: dict, NumPoints: int = 2) -> Tuple[dict, dict]:
    """Perform multi-point crossover on prompt dictionaries.
    
    Parameters
    ----------
    ParentOne, ParentTwo : dict
        Parent prompts
    NumPoints : int
        Number of crossover points
        
    Returns
    -------
    tuple
        Two child prompts
    """
    Points = sorted(random.sample(range(len(PromptSkeletonKeys)), min(NumPoints, len(PromptSkeletonKeys)-1)))
    Points = [0] + Points + [len(PromptSkeletonKeys)]
    
    ChildOne = {}
    ChildTwo = {}
    
    for i in range(len(Points) - 1):
        StartIdx = Points[i]
        EndIdx = Points[i + 1]
        
        if i % 2 == 0:
            for j in range(StartIdx, EndIdx):
                ChildOne[PromptSkeletonKeys[j]] = ParentOne.get(PromptSkeletonKeys[j], "")
                ChildTwo[PromptSkeletonKeys[j]] = ParentTwo.get(PromptSkeletonKeys[j], "")
        else:
            for j in range(StartIdx, EndIdx):
                ChildOne[PromptSkeletonKeys[j]] = ParentTwo.get(PromptSkeletonKeys[j], "")
                ChildTwo[PromptSkeletonKeys[j]] = ParentOne.get(PromptSkeletonKeys[j], "")
    
    ChildOne["Text"] = ParentOne.get("Text", "")
    ChildTwo["Text"] = ParentTwo.get("Text", "")
    
    return ChildOne, ChildTwo


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


def GetPromptHash(Prompt: dict) -> str:
    """Generate a hash for a prompt to use as cache key."""
    PromptStr = json.dumps(Prompt, sort_keys=True)
    return hashlib.md5(PromptStr.encode()).hexdigest()


def BatchApplyPromptToData(DataFrame: pd.DataFrame, Prompts: List[dict], 
                          BatchSize: int = 5, SampleSize: Optional[int] = None) -> List[pd.DataFrame]:
    """Apply multiple prompts to data in batches for efficiency.
    
    Parameters
    ----------
    DataFrame : pd.DataFrame
        Contains feedback and classification columns
    Prompts : List[dict]
        List of prompts to evaluate
    BatchSize : int
        Number of prompts to evaluate together
    SampleSize : Optional[int]
        If provided, only evaluate on a sample of the data
        
    Returns
    -------
    List[pd.DataFrame]
        List of result DataFrames, one per prompt
    """
    # Sample data if requested
    if SampleSize and SampleSize < len(DataFrame):
        DataFrame = DataFrame.sample(n=SampleSize, random_state=42)
    
    Results = []
    
    # Process prompts in batches
    for i in range(0, len(Prompts), BatchSize):
        BatchPrompts = Prompts[i:i+BatchSize]
        BatchResults = []
        
        # Create messages for all prompts and feedback combinations
        AllMessages = []
        PromptIndices = []
        
        for PromptIdx, Prompt in enumerate(BatchPrompts):
            PromptText = CombineString(Prompt)
            for _, Row in DataFrame.iterrows():
                FilledPrompt = PromptText.replace("{Text}", Row["feedback"])
                AllMessages.append([
                    {"role": "system", "content": "You are a helpful classification assistant."},
                    {"role": "user", "content": FilledPrompt}
                ])
                PromptIndices.append(PromptIdx)
        
        # Make concurrent API calls
        with ThreadPoolExecutor(max_workers=10) as Executor:
            Futures = []
            for Messages in AllMessages:
                Future = Executor.submit(
                    client.chat.completions.create,
                    model='gpt-4o',
                    messages=Messages,
                    temperature=0
                )
                Futures.append(Future)
            
            # Collect responses
            Responses = []
            for Future in as_completed(Futures):
                try:
                    Response = Future.result()
                    Responses.append(Response.choices[0].message.content.strip())
                except Exception as e:
                    Responses.append("Error")
        
        # Organize responses by prompt
        for PromptIdx in range(len(BatchPrompts)):
            PromptPredictions = []
            for i, Idx in enumerate(PromptIndices):
                if Idx == PromptIdx:
                    PromptPredictions.append(Responses[i])
            
            ResultDf = DataFrame.copy()
            ResultDf["Prediction"] = PromptPredictions
            BatchResults.append(ResultDf)
        
        Results.extend(BatchResults)
    
    return Results


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


def CalculateDiversity(Population: List[dict]) -> float:
    """Calculate diversity score for a population of prompts.
    
    Uses simple string distance between prompts.
    """
    if len(Population) < 2:
        return 1.0
    
    TotalDistance = 0
    Comparisons = 0
    
    for i in range(len(Population)):
        for j in range(i + 1, len(Population)):
            PromptA = CombineString(Population[i])
            PromptB = CombineString(Population[j])
            
            # Simple character-level difference
            Distance = sum(1 for a, b in zip(PromptA, PromptB) if a != b)
            Distance += abs(len(PromptA) - len(PromptB))
            
            TotalDistance += Distance
            Comparisons += 1
    
    return TotalDistance / (Comparisons * 100) if Comparisons else 0


def TournamentSelection(Population: List[dict], Scores: List[float], 
                       TournamentSize: int = 3) -> dict:
    """Select a parent using tournament selection.
    
    Parameters
    ----------
    Population : List[dict]
        Current population
    Scores : List[float]
        Fitness scores for each individual
    TournamentSize : int
        Number of individuals in each tournament
        
    Returns
    -------
    dict
        Selected parent prompt
    """
    Tournament = random.sample(list(zip(Population, Scores)), TournamentSize)
    Winner = max(Tournament, key=lambda x: x[1])
    return Winner[0]


def GeneratePopulation(PopulationSize: int) -> list:
    """Create an initial population of prompts."""
    return [GeneratePromptSample() for _ in range(PopulationSize)]


def MutatePrompt(Prompt: dict, MutationRate: float) -> dict:
    """Mutate fields within ``Prompt`` with probability ``MutationRate``."""
    for Field in PromptSkeletonKeys:
        if random.random() < MutationRate:
            MutatePromptField(Prompt, Field)
    return Prompt


def EvaluatePopulationBatch(Population: List[dict], DataFrame: pd.DataFrame, 
                           UseCache: bool = True, SampleSize: Optional[int] = None) -> List[float]:
    """Evaluate entire population efficiently using batching and caching."""
    
    # Separate cached and uncached prompts
    CachedScores = {}
    UncachedPrompts = []
    UncachedIndices = []
    
    for i, Prompt in enumerate(Population):
        PromptHash = GetPromptHash(Prompt)
        if UseCache and PromptHash in FitnessCache:
            CachedScores[i] = FitnessCache[PromptHash]
        else:
            UncachedPrompts.append(Prompt)
            UncachedIndices.append(i)
    
    # Evaluate uncached prompts in batch
    if UncachedPrompts:
        ResultFrames = BatchApplyPromptToData(DataFrame, UncachedPrompts, SampleSize=SampleSize)
        
        for Idx, (Prompt, ResultFrame) in enumerate(zip(UncachedPrompts, ResultFrames)):
            Score = CalculateFitnessScore(ResultFrame)
            OriginalIdx = UncachedIndices[Idx]
            
            if UseCache:
                PromptHash = GetPromptHash(Prompt)
                FitnessCache[PromptHash] = Score
            
            CachedScores[OriginalIdx] = Score
    
    # Return scores in original order
    return [CachedScores[i] for i in range(len(Population))]


def AdaptiveMutationRate(Generation: int, MaxGeneration: int, 
                        Diversity: float, BaseRate: float = 0.1) -> float:
    """Calculate adaptive mutation rate based on generation and diversity.
    
    Increases mutation when diversity is low or in later generations.
    """
    GenerationFactor = Generation / MaxGeneration
    DiversityFactor = 1.0 - Diversity
    
    AdaptiveRate = BaseRate * (1 + GenerationFactor + DiversityFactor)
    return min(AdaptiveRate, 0.5)  # Cap at 50%


def SaveCheckpoint(Population: List[dict], Generation: int, Stats: List[dict], 
                  Filename: str = "evolution_checkpoint.json"):
    """Save evolution state to file."""
    Checkpoint = {
        "generation": Generation,
        "population": Population,
        "stats": Stats,
        "fitness_cache": dict(list(FitnessCache.items())[:1000])  # Limit cache size
    }
    
    with open(Filename, 'w') as f:
        json.dump(Checkpoint, f, indent=2)


def LoadCheckpoint(Filename: str = "evolution_checkpoint.json") -> Optional[dict]:
    """Load evolution state from file."""
    try:
        with open(Filename, 'r') as f:
            Checkpoint = json.load(f)
        
        # Restore fitness cache
        global FitnessCache
        FitnessCache.update(Checkpoint.get("fitness_cache", {}))
        
        return Checkpoint
    except FileNotFoundError:
        return None


def RunEvolution(DataFrame: pd.DataFrame, PopulationSize: int, Generations: int,
                 MutationRate: float = 0.1, CrossoverProbability: float = 0.5,
                 Elitism: int = 1, UseTournament: bool = True, 
                 UseAdaptiveMutation: bool = True, UseDynamicPopulation: bool = True,
                 SaveCheckpoints: bool = True, CheckpointInterval: int = 5,
                 InitialSampleSize: int = None, VerboseStats: bool = True) -> list:
    """Evolve prompts over multiple generations with enhanced strategies.
    
    Parameters
    ----------
    DataFrame : pd.DataFrame
        Training data
    PopulationSize : int
        Base population size
    Generations : int
        Number of generations
    MutationRate : float
        Base mutation rate
    CrossoverProbability : float
        Crossover probability
    Elitism : int
        Number of elite individuals to preserve
    UseTournament : bool
        Use tournament selection instead of random
    UseAdaptiveMutation : bool
        Adapt mutation rate based on diversity
    UseDynamicPopulation : bool
        Adjust population size dynamically
    SaveCheckpoints : bool
        Save progress periodically
    CheckpointInterval : int
        Generations between checkpoints
    InitialSampleSize : int
        Start with smaller data sample for speed
    VerboseStats : bool
        Print detailed statistics
        
    Returns
    -------
    list
        Final population of prompts
    """
    
    global GenerationStats
    GenerationStats = []
    
    # Check for existing checkpoint
    if SaveCheckpoints:
        Checkpoint = LoadCheckpoint()
        if Checkpoint:
            Population = Checkpoint["population"]
            StartGeneration = Checkpoint["generation"] + 1
            GenerationStats = Checkpoint["stats"]
            print(f"Resuming from generation {StartGeneration}")
        else:
            Population = GeneratePopulation(PopulationSize)
            StartGeneration = 0
    else:
        Population = GeneratePopulation(PopulationSize)
        StartGeneration = 0
    
    CurrentPopSize = PopulationSize
    CurrentSampleSize = InitialSampleSize or len(DataFrame)
    
    for Generation in range(StartGeneration, Generations):
        StartTime = time.time()
        
        # Evaluate population
        Scores = EvaluatePopulationBatch(Population, DataFrame, 
                                       SampleSize=CurrentSampleSize)
        
        # Calculate statistics
        MeanScore = np.mean(Scores)
        MaxScore = max(Scores)
        MinScore = min(Scores)
        StdScore = np.std(Scores)
        Diversity = CalculateDiversity(Population)
        
        # Adaptive parameters
        if UseAdaptiveMutation:
            CurrentMutationRate = AdaptiveMutationRate(Generation, Generations, 
                                                      Diversity, MutationRate)
        else:
            CurrentMutationRate = MutationRate
        
        # Dynamic population sizing
        if UseDynamicPopulation:
            if Diversity < 0.1:  # Low diversity, expand population
                CurrentPopSize = min(PopulationSize * 2, 50)
            elif Diversity > 0.5 and Generation > Generations // 2:  # High diversity late-stage
                CurrentPopSize = max(PopulationSize // 2, 10)
            else:
                CurrentPopSize = PopulationSize
        
        # Gradually increase sample size
        if InitialSampleSize and CurrentSampleSize < len(DataFrame):
            CurrentSampleSize = min(
                int(CurrentSampleSize * 1.5),
                len(DataFrame)
            )
        
        # Sort population by fitness
        ScoredPopulation = list(zip(Population, Scores))
        ScoredPopulation.sort(key=lambda Item: Item[1], reverse=True)
        
        # Elite selection
        NewPopulation = [Item[0] for Item in ScoredPopulation[:Elitism]]
        
        # Generate offspring
        while len(NewPopulation) < CurrentPopSize:
            # Parent selection
            if UseTournament:
                ParentOne = TournamentSelection(Population, Scores)
                ParentTwo = TournamentSelection(Population, Scores)
            else:
                # Fallback to weighted random selection based on fitness
                Weights = [Score / sum(Scores) for Score in Scores]
                ParentOne = random.choices(Population, weights=Weights)[0]
                ParentTwo = random.choices(Population, weights=Weights)[0]
            
            # Crossover
            if random.random() < CrossoverProbability:
                ChildOne, ChildTwo = MultiPointCrossover(ParentOne, ParentTwo)
                Children = [ChildOne, ChildTwo]
            else:
                Children = [ParentOne.copy()]
            
            # Mutation
            for Child in Children:
                Child = MutatePrompt(Child, CurrentMutationRate)
                NewPopulation.append(Child)
                if len(NewPopulation) >= CurrentPopSize:
                    break
        
        Population = NewPopulation[:CurrentPopSize]
        
        # Track statistics
        ElapsedTime = time.time() - StartTime
        Stats = {
            "generation": Generation,
            "mean_fitness": MeanScore,
            "max_fitness": MaxScore,
            "min_fitness": MinScore,
            "std_fitness": StdScore,
            "diversity": Diversity,
            "mutation_rate": CurrentMutationRate,
            "population_size": CurrentPopSize,
            "sample_size": CurrentSampleSize,
            "elapsed_time": ElapsedTime,
            "cache_hits": len([p for p in Population if GetPromptHash(p) in FitnessCache])
        }
        GenerationStats.append(Stats)
        
        # Print statistics
        if VerboseStats:
            print(f"\nGeneration {Generation}:")
            print(f"  Fitness: {MaxScore:.3f} (max), {MeanScore:.3f} (mean), {StdScore:.3f} (std)")
            print(f"  Diversity: {Diversity:.3f}")
            print(f"  Mutation Rate: {CurrentMutationRate:.3f}")
            print(f"  Population Size: {CurrentPopSize}")
            print(f"  Sample Size: {CurrentSampleSize}/{len(DataFrame)}")
            print(f"  Time: {ElapsedTime:.2f}s")
            print(f"  Cache Efficiency: {Stats['cache_hits']}/{CurrentPopSize}")
        
        # Early stopping check
        if len(GenerationStats) > 10:
            RecentScores = [s["max_fitness"] for s in GenerationStats[-10:]]
            if max(RecentScores) - min(RecentScores) < 0.01:
                print(f"\nEarly stopping at generation {Generation} - fitness plateau detected")
                break
        
        # Save checkpoint
        if SaveCheckpoints and (Generation + 1) % CheckpointInterval == 0:
            SaveCheckpoint(Population, Generation, GenerationStats)
            print(f"  Checkpoint saved at generation {Generation}")
    
    # Final evaluation on full dataset if we used sampling
    if InitialSampleSize and CurrentSampleSize < len(DataFrame):
        print("\nFinal evaluation on full dataset...")
        Scores = EvaluatePopulationBatch(Population, DataFrame, SampleSize=None)
        ScoredPopulation = list(zip(Population, Scores))
        ScoredPopulation.sort(key=lambda Item: Item[1], reverse=True)
        Population = [Item[0] for Item in ScoredPopulation]
    
    return Population


def GetEvolutionStats() -> List[dict]:
    """Get statistics from the last evolution run."""
    return GenerationStats


def PlotEvolutionProgress():
    """Plot evolution progress (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
        
        if not GenerationStats:
            print("No evolution statistics available")
            return
        
        Generations = [s["generation"] for s in GenerationStats]
        MaxFitness = [s["max_fitness"] for s in GenerationStats]
        MeanFitness = [s["mean_fitness"] for s in GenerationStats]
        Diversity = [s["diversity"] for s in GenerationStats]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Fitness plot
        ax1.plot(Generations, MaxFitness, 'b-', label='Max Fitness')
        ax1.plot(Generations, MeanFitness, 'g--', label='Mean Fitness')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness Score')
        ax1.set_title('Evolution Progress')
        ax1.legend()
        ax1.grid(True)
        
        # Diversity plot
        ax2.plot(Generations, Diversity, 'r-', label='Population Diversity')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Diversity Score')
        ax2.set_title('Population Diversity')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('evolution_progress.png')
        plt.show()
        
    except ImportError:
        print("Matplotlib not installed. Cannot plot progress.")


# For backward compatibility - redirect old function calls to new ones
def EvaluatePrompt(Prompt: dict, DataFrame: pd.DataFrame, UseCache: bool = True) -> float:
    """Evaluate a single prompt - for backward compatibility."""
    Scores = EvaluatePopulationBatch([Prompt], DataFrame, UseCache=UseCache)
    return Scores[0]


def Crossover(ParentOne: dict, ParentTwo: dict, Probability: float = 0.5) -> dict:
    """Legacy uniform crossover - now uses multi-point internally."""
    if random.random() < 0.5:
        ChildOne, _ = MultiPointCrossover(ParentOne, ParentTwo, NumPoints=1)
        return ChildOne
    else:
        ChildOne, ChildTwo = MultiPointCrossover(ParentOne, ParentTwo)
        return random.choice([ChildOne, ChildTwo])


def ApplyPromptToData(DataFrame: pd.DataFrame, Prompt: str) -> pd.DataFrame:
    """Legacy single prompt application - for backward compatibility."""
    PromptDict = {"prompt": Prompt, "Text": "{Text}"}
    Results = BatchApplyPromptToData(DataFrame, [PromptDict])
    return Results[0] if Results else DataFrame.copy()