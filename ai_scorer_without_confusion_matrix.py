import os
import json
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

# ---------------- CONFIG ----------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

INPUT_FILE = "sample_50.xlsx"
OUTPUT_FILE = "radiology_reports_with_predictions_all_conditions_fast.xlsx"
JSON_FILE = "config.json"

MODEL_NAME = "gemini-2.0-flash-001"
DELAY_BETWEEN_REPORTS = 2
MAX_RETRIES = 6

# ---------------- LOAD CONDITIONS ----------------
with open(JSON_FILE, "r", encoding="utf-8") as f:
    conditions_list = json.load(f)

CONDITIONS = [c["condition"] for c in conditions_list]

print(f" Loaded {len(CONDITIONS)} conditions from {JSON_FILE}")
print(f"   {', '.join(CONDITIONS[:5])}... (+{len(CONDITIONS)-5} more)\n")

# ---------------- LOAD DATA ----------------
df = pd.read_excel(INPUT_FILE)
required_cols = [
    "Findings (original radiologist report)",
    "Conclusions (original radiologist report)",
    "Findings (AI report)",
    "Conclusions (AI report)",
]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# ---------------- MODEL INIT ----------------
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0,
    google_api_key=API_KEY
)

# ---------------- PROMPT TEMPLATE ----------------
batch_prompt_template = ChatPromptTemplate.from_template("""
You are an expert veterinary radiologist.

Evaluate the following report for the presence of these conditions:

{conditions_with_descriptions}

For each condition, return strictly:
condition_name: Positive or Negative

Report Findings:
{findings}

Report Conclusion:
{conclusions}
""")

batch_chain = RunnableSequence(batch_prompt_template | llm)


# ---------------- DETECTION FUNCTION ----------------
def detect_all_conditions(findings, conclusions, max_retries=MAX_RETRIES):
    conditions_with_descriptions = "\n".join(
        [f"- {c['condition']}: {c['prompt']}" for c in conditions_list]
    )

    for attempt in range(max_retries):
        try:
            response = batch_chain.invoke({
                "conditions_with_descriptions": conditions_with_descriptions,
                "findings": str(findings),
                "conclusions": str(conclusions)
            })

            text = response.content.strip()
            result_dict = {}

            for line in text.splitlines():
                if ":" in line:
                    cond_name, val = line.split(":", 1)
                    cond_name = cond_name.strip().lower()
                    val = val.strip()
                    # Match to known condition
                    for c in CONDITIONS:
                        if cond_name in c.lower().replace("_", " "):
                            result_dict[c] = val
                            break

            # Fill missing
            for c in CONDITIONS:
                if c not in result_dict:
                    result_dict[c] = "Negative"
            return result_dict

        except Exception as e:
            msg = str(e)
            if "429" in msg or "quota" in msg.lower():
                print(f"Quota limit hit! Waiting 60s (attempt {attempt+1})...")
                time.sleep(60)
                continue
            else:
                print(f" Error: {e}")
                return {c: "Error" for c in CONDITIONS}

    return {c: "Error (max retries)" for c in CONDITIONS}


# ---------------- MAIN LOOP ----------------
progress = tqdm(total=len(df), desc="Processing Reports", ncols=100)

for idx, row in df.iterrows():
    # Skip completed
    if all(f"{c}_original" in df.columns and pd.notna(df.at[idx, f"{c}_original"]) for c in CONDITIONS):
        progress.update(1)
        continue

    results_orig = detect_all_conditions(
        row["Findings (original radiologist report)"],
        row["Conclusions (original radiologist report)"]
    )

    results_ai = detect_all_conditions(
        row["Findings (AI report)"],
        row["Conclusions (AI report)"]
    )

    for c in CONDITIONS:
        df.at[idx, f"{c}_original"] = results_orig[c]
        df.at[idx, f"{c}_ai"] = results_ai[c]

    # Save progress
    if (idx + 1) % 5 == 0:
        temp_file = OUTPUT_FILE.replace(".xlsx", "_partial.xlsx")
        df.to_excel(temp_file, index=False)
        print(f"\n Autosaved after {idx + 1} rows")

    time.sleep(DELAY_BETWEEN_REPORTS)
    progress.update(1)

progress.close()

df.to_excel(OUTPUT_FILE, index=False)
print(f"\n Completed {len(df)} reports. Saved to {OUTPUT_FILE}")
