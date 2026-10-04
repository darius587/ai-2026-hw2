import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI



BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CANDIDATES_DIR = DATA_DIR / "candidates"

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. "
        "Put your OpenRouter API key in .env"
    )

MODEL = "openai/gpt-5"

client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
)



def load_json(filename):
    with open(DATA_DIR / filename, "r", encoding="utf-8") as file:
        return json.load(file)


RUBRIC = load_json("candidate_rubric.json")



def load_candidates():
    candidates = []

    for path in sorted(CANDIDATES_DIR.glob("story-*.md")):
        text = path.read_text(encoding="utf-8")

        candidates.append(
            {
                "file": path.name,
                "story": text,
            }
        )

    return candidates


def build_prompt(candidate):
    return f"""
You are evaluating one scholarship candidate.

Use ONLY the candidate story and the official rubric.

Official rubric:
{json.dumps(RUBRIC, ensure_ascii=False, indent=2)}

Candidate file:
{candidate["file"]}

Candidate story:
{candidate["story"]}

Return ONLY valid JSON with exactly these fields:

{{
  "academic": 0,
  "research": 0,
  "experience": 0,
  "gpa_4_scale": null,
  "gpa_scale_stated": null,
  "published_outputs": [],
  "unpublished_outputs": [],
  "experience_months": 0,
  "ambiguities": []
}}

Important rules:

- academic, research, and experience must each be a score from 0 to 5.
- Do not calculate the weighted total.
- Do not choose a winner.
- Never estimate a missing GPA.
- A publication counts only if the story says published or accepted.
- "in preparation", "submitted", "under review", "planned"
  and "in press" are NOT published.
- Count experience in months.
- Overlapping periods count only once.
- Undated experience cannot be counted.
- If the story contradicts itself, do not resolve the contradiction.
  Put it in ambiguities and use null where required.
- Use null for gpa_4_scale when GPA is missing or contradictory.
"""


# ============================================================
# ASK MODEL
# ============================================================

def evaluate_candidate(candidate):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful scholarship evaluation assistant. "
                    "Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": build_prompt(candidate),
            },
        ],
        response_format={
            "type": "json_object"
        },
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            f"Empty response for {candidate['file']}"
        )

    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON for {candidate['file']}: {exc}"
        )

    return result



def validate_result(result):
    required = [
        "academic",
        "research",
        "experience",
        "gpa_4_scale",
        "gpa_scale_stated",
        "published_outputs",
        "unpublished_outputs",
        "experience_months",
        "ambiguities",
    ]

    for field in required:
        if field not in result:
            raise ValueError(
                f"Missing field: {field}"
            )

    for field in [
        "academic",
        "research",
        "experience",
    ]:
        value = result[field]

        if not isinstance(value, (int, float)):
            raise ValueError(
                f"{field} must be a number"
            )

        if not 0 <= value <= 5:
            raise ValueError(
                f"{field} must be between 0 and 5"
            )

    if not isinstance(
        result["published_outputs"], list
    ):
        raise ValueError(
            "published_outputs must be a list"
        )

    if not isinstance(
        result["unpublished_outputs"], list
    ):
        raise ValueError(
            "unpublished_outputs must be a list"
        )

    if not isinstance(
        result["experience_months"],
        (int, float),
    ):
        raise ValueError(
            "experience_months must be a number"
        )

    if not isinstance(
        result["ambiguities"],
        list
    ):
        raise ValueError(
            "ambiguities must be a list"
        )



def calculate_total(result):
    academic_weight = 0.5
    research_weight = 0.3
    experience_weight = 0.2

    total = (
        academic_weight * result["academic"]
        + research_weight * result["research"]
        + experience_weight * result["experience"]
    )

    return round(total, 2)



def main():
    print("=" * 80)
    print("HW2 - Sublab Hard")
    print("CV Extraction and Ranking")
    print("=" * 80)

    print(f"Model: {MODEL}")

    candidates = load_candidates()

    print(
        f"Candidates found: {len(candidates)}"
    )

    if not candidates:
        raise RuntimeError(
            "No candidate stories found."
        )

    results = []

    for candidate in candidates:
        print("\n" + "-" * 80)
        print(
            f"Evaluating: {candidate['file']}"
        )

        result = evaluate_candidate(candidate)

        validate_result(result)

        total = calculate_total(result)

        item = {
            "file": candidate["file"],
            "scores": result,
            "weighted_total": total,
        }

        results.append(item)

        print(
            f"Academic:   {result['academic']}"
        )
        print(
            f"Research:   {result['research']}"
        )
        print(
            f"Experience: {result['experience']}"
        )
        print(
            f"Total:      {total}"
        )


    results.sort(
        key=lambda item: item["weighted_total"],
        reverse=True,
    )

    print("\n" + "=" * 80)
    print("FINAL RANKING")
    print("=" * 80)

    for index, item in enumerate(
        results,
        start=1,
    ):
        print(
            f"{index}. "
            f"{item['file']} "
            f"- {item['weighted_total']:.2f}"
        )

    winner = results[0]

    print("\n" + "=" * 80)
    print("WINNER")
    print("=" * 80)

    print(
        f"{winner['file']} "
        f"with score {winner['weighted_total']:.2f}"
    )

    print("\nDetailed results:")

    print(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()