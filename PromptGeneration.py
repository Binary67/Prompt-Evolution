def GeneratePromptSample(RoleAssignment: str = "You are an expert in employee-feedback analysis.") -> str:
    """
    Calls Azure OpenAI once and returns a single, fully-formed prompt string
    (except for {Text}, which stays as a placeholder).

    Parameters
    ----------
    RoleAssignment : str, optional
        A custom persona you want at the top of the prompt.

    Returns
    -------
    str
        A ready-to-use prompt for classifying feedback into
        'Compliment' or 'Development'.
    """


    PromptSkeleton = """{RoleAssignment}
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

    return PromptResult
