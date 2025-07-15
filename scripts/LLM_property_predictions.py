# assay_gemini_flag.py
"""
Return 0 (not endocrine-relevant) or 1 (endocrine-relevant)
for every assay listed in assay_list.txt, using Google Gemini.
"""
import os, enum
from google import genai
from google.genai import types
from google.genai.types import Tool, GoogleSearch, GenerateContentConfig, Enum
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# --- 1. Define the only valid outputs -------------
# YnFlag = Enum(values=["0", "1"])
class YnFlag(enum.Enum):
    NO = "0"
    YES = "1"

# reusable search tool object
GOOGLE_SEARCH = Tool(google_search=GoogleSearch())

# --- 2. Init client --------------------------------
load_dotenv()  # Load environment variables from .env file
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL_ID = "gemini-2.5-flash"   # Fast & supports responseSchema

grounding_tool = types.Tool(google_search=types.GoogleSearch())
BASE_CFG = types.GenerateContentConfig(
    tools=[grounding_tool],
    response_schema=YnFlag,
    # response_mime_type="text/x.enum",
    temperature=0.0,
    top_p=0.1,
    # max_output_tokens=1,
)

# --- 3. Core helper --------------------------------
def flag_assay(assay: str, toxicity_type : str = "ed") -> int:
#     prompt = f"""
# You are a toxicology expert. Classify this assay as either developmental/reproductive toxicity (DART) relevant or not.

# DART-relevant (respond '1'): Assays that directly measure:
# - Embryonic/fetal development
# - Reproductive organ function
# - Fertility parameters
# - Developmental neurotoxicity specific to developing organisms

# NOT DART-relevant (respond '0'): Assays that measure:
# - General cytotoxicity
# - Metabolic disruption without developmental context
# - General inflammation
# - Adult-specific endpoints

# Assay Name: {assay}

# Response (0 or 1 only):"""
    if toxicity_type == "dart":
        prompt = (
            "You are a toxicology expert. Classify this assay as either "
            "developmental/reproductive toxicity (DART) relevant or not.\n\n"
            "DART-relevant (respond '1'): assays that directly measure:\n"
            "- Embryonic/fetal development\n"
            "- Reproductive organ function\n"
            "- Fertility parameters\n"
            "- Developmental neurotoxicity in developing organisms\n\n"
            "NOT DART-relevant (respond '0'): assays that measure:\n"
            "- General cytotoxicity\n"
            "- Metabolic disruption with no developmental context\n"
            "- General inflammation\n"
            "- Adult-only endpoints\n\n"
            f"Assay Name: {assay}\n\n"
            "Response (0 or 1 only):"
        )
    elif toxicity_type == "ed":
        prompt = (
            "You are a toxicology expert. Classify this assay as either "
            "endocrine disruption (ED) relevant or not.\n\n"
            "ED-relevant (respond '1'): assays that directly measure:\n"
            "- Estrogen, androgen, progesterone, or thyroid receptor binding/activation\n"
            "- Steroidogenesis or aromatase activity (e.g., alterations in testosterone, estradiol)\n"
            "- Hormone-regulated gene/protein expression (e.g., vitellogenin induction, uterotrophic response)\n"
            "- Developmental or reproductive endpoints mediated by endocrine pathways "
            "   (e.g., altered sex ratio, secondary sexual characteristics, thyroid histopathology)\n"
            "- Endocrine-axis hormone or biomarker levels in vivo (e.g., circulating T3/T4, LH/FSH)\n\n"
            "NOT ED-relevant (respond '0'): assays that measure:\n"
            "- General cytotoxicity or cell viability with no hormonal context\n"
            "- Mitochondrial dysfunction or metabolic stress unrelated to endocrine pathways\n"
            "- Generic inflammatory or immune responses without hormonal mediation\n"
            "- Adult-organ toxicity unlinked to endocrine mechanisms (e.g., hepatotoxicity, nephrotoxicity)\n"
            "- Non-specific oxidative stress, DNA damage, or genotoxicity endpoints\n\n"
            f"Assay Name: {assay}\n\n"
            "Response (0 or 1 only):"
        )

    resp = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=BASE_CFG,
        # config    = GenerateContentConfig(
        #     tools           = [GOOGLE_SEARCH],   # search grounding
        #     response_schema = YnFlag,            # enforces enum
        #     temperature     = 0.0,
        #     top_p           = 0.1,
        #     max_output_tokens = 1,
        # ),
        # config={
        #     "response_mime_type": "text/x.enum",
        #     "response_schema": YnFlag,   # enum schema with '0' and '1'
        #     "temperature": 0.0,
        #     "top_p": 0.1,
        #     # "max_output_tokens": 1,  # risk of truncation
        # },
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
