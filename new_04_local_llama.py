import pandas as pd
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ---------------- CONFIG ----------------
INPUT_FILE = "sample.xlsx"
OUTPUT_FILE = "radiology_reports_with_predictions_one_conditions_llama.xlsx"

MODEL_PATH = (
    r"C:\Users\Admin\Downloads\Vetology\research_app"
    r"\llama-3-Korean-Bllossom-8B-Q4_K_M.gguf"
)

# LOAD MODEL (FASTER SETTINGS)
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.1,
    n_ctx=2048,        # smaller context = faster
    n_batch=128,       # better batching
    n_threads=8,       # set to number of CPU cores
    verbose=False,
    streaming=False
)

# PROMPT TEMPLATE
prompt_template = """
You are an expert veterinary radiologist.
Determine if the following report indicates **{condition}**.

Return ONLY one word: Positive or Negative.
Do not explain. Decide quickly.

Findings:
{findings}

Conclusions:
{conclusions}
"""

prompt = PromptTemplate(
    input_variables=["condition", "findings", "conclusions"],
    template=prompt_template,
)

chain = LLMChain(llm=llm, prompt=prompt)

# CONDITIONS
canine_conditions = {
"bronchitis": {
    "name": "bronchitis",
    "options": ["Positive", "Negative"],
    "prompt": """
You are a veterinary radiology classifier.
Determine whether this report shows evidence of bronchitis.

Return only one:
Positive — bronchitis present
Negative — no bronchitis seen

Respond with one word only.
"""
}
}

# LOAD DATA
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


# DETECTION FUNCTION
def detect_condition(findings, conclusions, condition):
    try:
        resp = chain.invoke({
            "condition": condition,
            "findings": str(findings),
            "conclusions": str(conclusions)
        })
        text = resp["text"].strip()

        if "positive" in text.lower():
            return "Positive"
        if "negative" in text.lower():
            return "Negative"
        return "Unknown"
    except (KeyboardInterrupt, SystemExit):
        # allow program interruption to propagate
        raise
    except Exception as e:
        # log unexpected errors and return Unknown
        print(f"Error detecting {condition}: {e}")
        return "Unknown"


# MAIN LOOP
for condition in canine_conditions.keys():
    print(f"Evaluating: {condition}")

    df[f"{condition}_original"] = ""
    df[f"{condition}_ai"] = ""

    for idx, row in df.iterrows():
        df.at[idx, f"{condition}_original"] = detect_condition(
            row["Findings (original radiologist report)"],
            row["Conclusions (original radiologist report)"],
            condition
        )

        df.at[idx, f"{condition}_ai"] = detect_condition(
            row["Findings (AI report)"],
            row["Conclusions (AI report)"],
            condition
        )

# SAVE
df.to_excel(OUTPUT_FILE, index=False)
print(f"Completed! Results saved to: {OUTPUT_FILE}")
