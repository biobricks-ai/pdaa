# assay_gemini_flag.py
"""
Return 0 (not endocrine-relevant) or 1 (endocrine-relevant)
for every assay listed in assay_list.txt, using Google Gemini.
"""
import os, enum
from google import genai
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# --- 1. Define the only valid outputs -------------
class YnFlag(enum.Enum):
    NO = "0"
    YES = "1"

# --- 2. Init client --------------------------------
load_dotenv()  # Load environment variables from .env file
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL_ID = "gemini-2.5-flash"   # Fast & supports responseSchema

# --- 3. Core helper --------------------------------
def flag_assay(assay: str) -> int:
    prompt = f"""
Analyze the following toxicology assay and determine if it is a primary measure of developmental and reproductive toxicity (DART).
- Respond with '0' if it measures more general endpoints like cytotoxicity, metabolic disruption, or inflammation, which are not primary DART endpoints.
- Respond with '1' if it directly assesses endpoints like teratogenicity, reproductive organ function, fertility, or developmental neurotoxicity.

Assay Name: {assay}
    """
    # prompt = "Respond with 0 (No) to this question"  # this works, returning 0
    resp = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        # Everything below enforces exact output --------------------------
        config={
            # "response_mime_type": "text/x.enum",
            # "response_schema": YnFlag,   # enum schema with '0' and '1'
            "temperature": 0.0,
            "top_p": 0.4,
            # "max_output_tokens": 1,
        },
    )
    return resp.text.strip()
    # return int(resp.text.strip())

# --- 4. Batch run ----------------------------------
if __name__ == "__main__":
    resource_dir = Path("resources")
    assay_file = resource_dir / "assay_list.txt"
    with open(assay_file) as fh:
        assays = [ln.strip() for ln in fh if ln.strip()]

    results = {}
    for assay in tqdm(assays, desc="Flagging assays"):
        results[assay] = flag_assay(assay)

    # save results to a file
    output_file = resource_dir / "assay_flags.txt"
    with open(output_file, "w") as fh:
        for assay, flag in results.items():
            fh.write(f"{assay}\t{flag}\n")
