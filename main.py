import pandas as pd
from PromptGeneration import (
    RunEvolution, EvaluatePrompt, CombineString, 
    GetEvolutionStats, PlotEvolutionProgress,
    EvaluatePopulationBatch
)
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

# Sample dataset for demonstration
# In production, load your full dataset here
SampleData = pd.DataFrame([
    {"feedback": "Great job on the project presentation", "classification": "compliment"},
    {"feedback": "Your code quality is excellent", "classification": "compliment"},
    {"feedback": "Needs improvement in communication skills", "classification": "development"},
    {"feedback": "Consider being more proactive in meetings", "classification": "development"},
    {"feedback": "Outstanding problem-solving abilities", "classification": "compliment"},
    {"feedback": "Should work on time management", "classification": "development"},
])

# Evolution configuration
PopulationSize = 10  # Increased for better diversity
Generations = 5     # More generations for demonstration

# Run evolution with enhanced features
print("Starting enhanced evolution process...")
print("=" * 60)

FinalPopulation = RunEvolution(
    SampleData, 
    PopulationSize, 
    Generations,
    MutationRate=0.1,
    CrossoverProbability=0.7,
    Elitism=2,                    # Keep top 2 performers
    UseTournament=True,           # Tournament selection
    UseAdaptiveMutation=True,     # Adaptive mutation rates
    UseDynamicPopulation=True,    # Dynamic population sizing
    SaveCheckpoints=True,         # Save progress
    CheckpointInterval=5,         # Save every 5 generations
    InitialSampleSize=4,          # Start with smaller sample for speed
    VerboseStats=True            # Print detailed statistics
)

print("\n" + "=" * 60)
print("Evolution complete! Analyzing final population...")
print("=" * 60)

# Evaluate final population on full dataset
FinalScores = EvaluatePopulationBatch(FinalPopulation, SampleData, UseCache=True)

# Get best prompt
BestIndex = max(range(len(FinalScores)), key=FinalScores.__getitem__)
BestPrompt = FinalPopulation[BestIndex]
BestScore = FinalScores[BestIndex]

# Display results
print(f"\nTop 3 Prompts:")
print("-" * 60)

SortedResults = sorted(zip(FinalPopulation, FinalScores), key=lambda x: x[1], reverse=True)
for i, (Prompt, Score) in enumerate(SortedResults[:3]):
    print(f"\n{i+1}. Fitness Score: {Score:.3f}")
    print(f"   First 200 chars: {CombineString(Prompt)[:200]}...")

print("\n" + "=" * 60)
print("BEST PROMPT:")
print("=" * 60)
print(CombineString(BestPrompt))
print(f"\nFinal Fitness Score: {BestScore:.3f}")

# Display evolution statistics
Stats = GetEvolutionStats()
if Stats:
    print("\n" + "=" * 60)
    print("EVOLUTION SUMMARY:")
    print("=" * 60)
    
    InitialMax = Stats[0]["max_fitness"]
    FinalMax = Stats[-1]["max_fitness"]
    TotalTime = sum(s["elapsed_time"] for s in Stats)
    CacheEfficiency = sum(s["cache_hits"] for s in Stats) / sum(s["population_size"] for s in Stats)
    
    print(f"Initial Best Fitness: {InitialMax:.3f}")
    print(f"Final Best Fitness: {FinalMax:.3f}")
    print(f"Improvement: {((FinalMax - InitialMax) / InitialMax * 100):.1f}%")
    print(f"Total Evolution Time: {TotalTime:.1f}s")
    print(f"Average Time per Generation: {TotalTime/len(Stats):.1f}s")
    print(f"Cache Efficiency: {CacheEfficiency:.1%}")
    
    # Check if early stopping occurred
    if len(Stats) < Generations:
        print(f"Early stopping at generation {len(Stats)-1}")

# Try to plot progress (if matplotlib is available)
print("\nAttempting to generate evolution progress plot...")
PlotEvolutionProgress()

print("\nEvolution process complete!")