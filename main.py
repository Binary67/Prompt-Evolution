import pandas as pd
from PromptGeneration import RunEvolution, EvaluatePrompt, CombineString, ClassificationTaskConfig
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

###################################################
### Example 1: Employee Feedback Classification ###
###################################################

print("Example 1: Employee Feedback Classification")
print("-"*50)

FeedbackData = pd.DataFrame([
    {"feedback": "Great job on the project", "classification": "compliment"},
    {"feedback": "Needs improvement in communication", "classification": "development"},
    {"feedback": "Excellent presentation skills", "classification": "compliment"},
    {"feedback": "Consider working on time management", "classification": "development"},
])

# Configure for feedback classification
FeedbackConfig = ClassificationTaskConfig(
    Labels=["compliment", "development"],
    TaskDescription="Classify whether the following employee feedback is a Compliment or Development feedback:",
    DataColumnName="feedback",
    LabelColumnName="classification"
)

PopulationSize = 4
Generations = 2

FinalPopulation = RunEvolution(FeedbackData, FeedbackConfig, PopulationSize, Generations)
Scores = [EvaluatePrompt(Prompt, FeedbackData, FeedbackConfig) for Prompt in FinalPopulation]
BestIndex = max(range(len(Scores)), key=Scores.__getitem__)
BestPrompt = FinalPopulation[BestIndex]
BestScore = Scores[BestIndex]

print("Best Prompt:\n", CombineString(BestPrompt))
print("Fitness Score:", BestScore)

#####################################
### Example 2: Sentiment Analysis ###
#####################################

print("\n\nExample 2: Sentiment Analysis")
print("-"*50)

SentimentData = pd.DataFrame([
    {"text": "I love this product! It's amazing!", "label": "positive"},
    {"text": "This is terrible, worst purchase ever", "label": "negative"},
    {"text": "It's okay, nothing special", "label": "neutral"},
    {"text": "Absolutely fantastic experience", "label": "positive"},
])

# Configure for sentiment analysis
SentimentConfig = ClassificationTaskConfig(
    Labels=["positive", "negative", "neutral"],
    DataColumnName="text",
    LabelColumnName="label"
)

print("Running evolution for sentiment analysis...")
FinalPopulation = RunEvolution(SentimentData, SentimentConfig, PopulationSize, Generations)
Scores = [EvaluatePrompt(Prompt, SentimentData, SentimentConfig) for Prompt in FinalPopulation]
BestIndex = max(range(len(Scores)), key=Scores.__getitem__)
BestScore = Scores[BestIndex]

print(f"Best fitness score for sentiment analysis: {BestScore}")

# Example 3: Email Spam Classification
print("\n\nExample 3: Email Spam Classification")
print("-"*50)

SpamData = pd.DataFrame([
    {"email_text": "You've won $1000! Click here to claim", "category": "spam"},
    {"email_text": "Meeting scheduled for tomorrow at 2pm", "category": "ham"},
    {"email_text": "URGENT: Verify your account now!!!", "category": "spam"},
    {"email_text": "Please review the attached document", "category": "ham"},
])

# Configure for spam classification
SpamConfig = ClassificationTaskConfig(
    Labels=["spam", "ham"],
    TaskDescription="Determine if the following email is spam or ham (legitimate):",
    DataColumnName="email_text",
    LabelColumnName="category"
)

print("Running evolution for spam classification...")
FinalPopulation = RunEvolution(SpamData, SpamConfig, PopulationSize, Generations)
Scores = [EvaluatePrompt(Prompt, SpamData, SpamConfig) for Prompt in FinalPopulation]
BestIndex = max(range(len(Scores)), key=Scores.__getitem__)
BestScore = Scores[BestIndex]

print(f"Best fitness score for spam classification: {BestScore}")
