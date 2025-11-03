import os
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

# CONFIG
# Load .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
INPUT_FILE = "sample.xlsx"
OUTPUT_FILE = "radiology_reports_with_predictions_all_conditions_up.xlsx"
JSON_FILE = "config.json"

#  LOAD CONDITIONS JSON
with open(JSON_FILE, "r") as f:
    canine_conditions = json.load(f)

#  LOAD EXCEL
df = pd.read_excel(INPUT_FILE)

required_cols = [
    "Findings (original radiologist report)",
    "Conclusions (original radiologist report)",
    "Findings (AI report)",
    "Conclusions (AI report)",
]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

#  MODEL INIT
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",  # gemini-2.0-flash-001
    temperature=0.1,
)


def make_chain(system_prompt):
    """Create a LangChain pipeline for each condition prompt."""
    template = ChatPromptTemplate.from_template(
        system_prompt + """

Report Findings:
{findings}

Report Conclusion:
{conclusions}
"""
    )
    return RunnableSequence(template | llm)


#  DETECTION FUNCTION
def detect_condition(findings, conclusions, prompt_text):
    chain = make_chain(prompt_text)
    try:
        response = chain.invoke(
            {
                "findings": str(findings),
                "conclusions": str(conclusions),
            }
        )
        return response.content.strip()
    except Exception as err:
        print(f"Error: {err}")
        return "Unknown"


#  LOOP THROUGH CONDITIONS
for cond in canine_conditions:
    condition_name = cond["condition"]
    condition_prompt = cond["prompt"]

    print(f"Evaluating condition: {condition_name}")

    df[f"{condition_name}_original"] = df.apply(
        lambda row, cp=condition_prompt: detect_condition(
            row["Findings (original radiologist report)"],
            row["Conclusions (original radiologist report)"],
            cp,
        ),
        axis=1,
    )

    df[f"{condition_name}_ai"] = df.apply(
        lambda row, cp=condition_prompt: detect_condition(
            row["Findings (AI report)"],
            row["Conclusions (AI report)"],
            cp,
        ),
        axis=1,
    )

#  SAVE OUTPUT
df.to_excel(OUTPUT_FILE, index=False)
print(f"All conditions processed & saved to: {OUTPUT_FILE}")
