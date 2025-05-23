import pandas as pd
import asyncio
from PromptGeneration import RunEvolution, EvaluatePrompt, CombineString, ClassificationTaskConfig

async def main():
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

    FinalPopulation = await RunEvolution(FeedbackData, FeedbackConfig, PopulationSize, Generations)
    
    # Evaluate final population in parallel
    EvaluationTasks = [EvaluatePrompt(Prompt, FeedbackData, FeedbackConfig) for Prompt in FinalPopulation]
    Scores = await asyncio.gather(*EvaluationTasks)
    
    BestIndex = max(range(len(Scores)), key=Scores.__getitem__)
    BestPrompt = FinalPopulation[BestIndex]
    BestScore = Scores[BestIndex]

    print("\nBest Prompt:\n", CombineString(BestPrompt))
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
    FinalPopulation = await RunEvolution(SentimentData, SentimentConfig, PopulationSize, Generations)
    
    # Evaluate final population in parallel
    EvaluationTasks = [EvaluatePrompt(Prompt, SentimentData, SentimentConfig) for Prompt in FinalPopulation]
    Scores = await asyncio.gather(*EvaluationTasks)
    
    BestIndex = max(range(len(Scores)), key=Scores.__getitem__)
    BestScore = Scores[BestIndex]

    print(f"\nBest fitness score for sentiment analysis: {BestScore}")

    ############################################
    ### Example 3: Email Spam Classification ###
    ############################################
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
    FinalPopulation = await RunEvolution(SpamData, SpamConfig, PopulationSize, Generations)
    
    # Evaluate final population in parallel
    EvaluationTasks = [EvaluatePrompt(Prompt, SpamData, SpamConfig) for Prompt in FinalPopulation]
    Scores = await asyncio.gather(*EvaluationTasks)
    
    BestIndex = max(range(len(Scores)), key=Scores.__getitem__)
    BestScore = Scores[BestIndex]

    print(f"\nBest fitness score for spam classification: {BestScore}")

if __name__ == "__main__":
    asyncio.run(main())