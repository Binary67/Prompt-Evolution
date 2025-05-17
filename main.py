import pandas as pd
from PromptGeneration import RunEvolution, EvaluatePrompt, CombineString

# Sample dataset for demonstration
SampleData = pd.DataFrame([
    {"feedback": "Great job on the project", "classification": "compliment"},
    {"feedback": "Needs improvement in communication", "classification": "development"},
])

PopulationSize = 4
Generations = 2

FinalPopulation = RunEvolution(SampleData, PopulationSize, Generations)
Scores = [EvaluatePrompt(Prompt, SampleData) for Prompt in FinalPopulation]
BestIndex = max(range(len(Scores)), key=Scores.__getitem__)
BestPrompt = FinalPopulation[BestIndex]
BestScore = Scores[BestIndex]

print("Best Prompt:\n", CombineString(BestPrompt))
print("Fitness Score:", BestScore)
