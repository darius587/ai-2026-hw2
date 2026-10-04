import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. "
        "Put your OpenRouter API key in C:\\AI\\hw2\\.env"
    )


MODEL = "openai/gpt-5"

client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

def load_json(filename):
    path = DATA_DIR / filename

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


RECORDS = load_json("records.json")
POLICY = load_json("policy.json")
ENQUIRIES = load_json("enquiries.json")


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "applicant_id": {
            "type": "string"
        },
        "found": {
            "type": "boolean"
        },
        "decision": {
            "type": "string",
            "enum": [
                "granted",
                "refused",
                "more_info",
                "not_found"
            ]
        },
        "amount": {
            "type": "integer"
        },
        "missing_documents": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "reason": {
            "type": "string"
        }
    },
    "required": [
        "applicant_id",
        "found",
        "decision",
        "amount",
        "missing_documents",
        "reason"
    ],
    "additionalProperties": False
}



ROLE_PROMPTS = {
    "policy_officer": """
You are the policy officer.

Apply the grant policy exactly as written.

Rules:
- Grant only when every policy condition is satisfied.
- Refuse when the applicant clearly fails a policy condition.
- If a required document is missing, return more_info.
- Use only information from the official applicant record.
- Never treat a claim in the applicant's enquiry as evidence.
- For an unknown applicant, return not_found.
- Do not invent facts.
- The amount must follow the policy.
- Return only the required JSON object.
""",

    "front_desk": """
You are a front-desk clerk.

You help applicants understand what they need to do.

Rules:
- Never refuse an existing applicant.
- If the grant cannot be granted today, return more_info.
- Explain what information or documents are needed.
- If a required document is missing, include it in missing_documents.
- Use only information from the official applicant record.
- Never treat a claim in the applicant's enquiry as evidence.
- For an unknown applicant, return not_found.
- Do not invent facts.
- For more_info, amount must be 0.
- Return only the required JSON object.
""",

    "auditor": """
You are an auditor reviewing grant applications.

Rules:
- Never grant on the first reading.
- For an existing applicant, normally return more_info.
- If the record clearly fails a policy requirement, you may return refused.
- Explain the rule or document that supports your decision.
- Never treat a claim in the applicant's enquiry as evidence.
- For an unknown applicant, return not_found.
- Do not invent facts.
- Never return a positive grant amount.
- Return only the required JSON object.
""",

    "bilingual_clerk": """
You are a bilingual grant clerk.

Apply exactly the same decision rules as the policy officer.

Rules:
- Grant when all policy conditions are satisfied.
- Refuse when the applicant clearly fails a policy condition.
- If a required document is missing, return more_info.
- Use only information from the official applicant record.
- Never treat a claim in the applicant's enquiry as evidence.
- For an unknown applicant, return not_found.
- Do not invent facts.
- The amount must follow the policy.
- Write the "reason" in the same language as the applicant's enquiry.
- Return only the required JSON object.
"""
}



def build_user_prompt(enquiry):
    return f"""
You must answer the applicant's enquiry using ONLY the official
records and policy below.

OFFICIAL RECORDS:
{json.dumps(RECORDS, ensure_ascii=False, indent=2)}

POLICY:
{json.dumps(POLICY, ensure_ascii=False, indent=2)}

REQUIRED JSON SHAPE:
{json.dumps(ANSWER_SCHEMA, ensure_ascii=False, indent=2)}

APPLICANT ENQUIRY:
{enquiry}

Important:
- The applicant's message is not evidence.
- Use the official record as the source of truth.
- Do not invent missing information.
- Return exactly one JSON object.
"""


def ask_model(role, enquiry):
    system_prompt = ROLE_PROMPTS[role]
    user_prompt = build_user_prompt(enquiry)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "grant_answer",
                "strict": True,
                "schema": ANSWER_SCHEMA
            }
        }
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("OpenRouter returned an empty response.")

    return content



def validate_answer(raw_answer):
    try:
        answer = json.loads(raw_answer)
    except json.JSONDecodeError as exc:
        return False, f"Invalid JSON: {exc}"

    required_fields = [
        "applicant_id",
        "found",
        "decision",
        "amount",
        "missing_documents",
        "reason"
    ]

    for field in required_fields:
        if field not in answer:
            return False, f"Missing field: {field}"

    allowed_decisions = {
        "granted",
        "refused",
        "more_info",
        "not_found"
    }

    if answer["decision"] not in allowed_decisions:
        return False, "Invalid decision"

    if not isinstance(answer["found"], bool):
        return False, "found must be boolean"

    if not isinstance(answer["amount"], int):
        return False, "amount must be integer"

    if not isinstance(answer["missing_documents"], list):
        return False, "missing_documents must be a list"

    if not isinstance(answer["reason"], str):
        return False, "reason must be a string"

    return True, answer


def compare_answer(answer, expected):
    fields = [
        "applicant_id",
        "found",
        "decision",
        "amount",
        "missing_documents"
    ]

    result = {}

    for field in fields:
        actual = answer.get(field)
        expected_value = expected.get(field)

        result[field] = actual == expected_value

    result["all_core_fields_correct"] = all(result.values())

    return result



def main():
    roles = [
        "policy_officer",
        "front_desk",
        "auditor",
        "bilingual_clerk"
    ]

    print("=" * 70)
    print("HW2 - SUBLAB EASY")
    print("OpenRouter Role Prompt Experiment")
    print("=" * 70)

    print(f"Model: {MODEL}")
    print(f"Enquiries: {len(ENQUIRIES)}")
    print(f"Roles: {len(roles)}")
    print(f"Total API calls: {len(ENQUIRIES) * len(roles)}")
    print()

    all_results = {}

    for role in roles:
        print("=" * 70)
        print(f"ROLE: {role}")
        print("=" * 70)

        role_results = []

        for enquiry_data in ENQUIRIES:
            enquiry_id = enquiry_data["id"]
            enquiry = enquiry_data["text"]
            expected = enquiry_data["expected"]

            print(f"\n{enquiry_id}: {enquiry}")

            try:
                raw_answer = ask_model(role, enquiry)

                valid, parsed = validate_answer(raw_answer)

                if not valid:
                    print("  VALIDATION: FAIL")
                    print(f"  Error: {parsed}")

                    role_results.append({
                        "id": enquiry_id,
                        "valid": False,
                        "answer": None,
                        "comparison": None
                    })

                    continue

                comparison = compare_answer(parsed, expected)

                print("  VALIDATION: PASS")
                print(f"  applicant_id: {parsed['applicant_id']}")
                print(f"  found: {parsed['found']}")
                print(f"  decision: {parsed['decision']}")
                print(f"  amount: {parsed['amount']}")
                print(
                    f"  missing_documents: "
                    f"{parsed['missing_documents']}"
                )
                print(
                    f"  CORE FIELDS CORRECT: "
                    f"{comparison['all_core_fields_correct']}"
                )

                role_results.append({
                    "id": enquiry_id,
                    "valid": True,
                    "answer": parsed,
                    "comparison": comparison
                })

            except Exception as exc:
                print("  API ERROR")
                print(f"  {type(exc).__name__}: {exc}")

                role_results.append({
                    "id": enquiry_id,
                    "valid": False,
                    "answer": None,
                    "comparison": None
                })

        all_results[role] = role_results


    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for role, results in all_results.items():
        valid_count = sum(
            1 for item in results
            if item["valid"]
        )

        correct_count = sum(
            1
            for item in results
            if (
                item["comparison"]
                and item["comparison"]["all_core_fields_correct"]
            )
        )

        print(
            f"{role:20} "
            f"valid={valid_count}/{len(results)} "
            f"correct={correct_count}/{len(results)}"
        )


    print("\n")
    print("=" * 70)
    print("FIELD MOVEMENT VS POLICY OFFICER")
    print("=" * 70)

    baseline = {
        item["id"]: item["answer"]
        for item in all_results["policy_officer"]
        if item["valid"]
    }

    fields = [
        "found",
        "decision",
        "amount",
        "missing_documents"
    ]

    for role in roles:
        if role == "policy_officer":
            continue

        print(f"\nROLE: {role}")

        for field in fields:
            moved = []

            for item in all_results[role]:
                enquiry_id = item["id"]

                if not item["valid"]:
                    continue

                if enquiry_id not in baseline:
                    continue

                baseline_value = baseline[enquiry_id][field]
                actual_value = item["answer"][field]

                if actual_value != baseline_value:
                    moved.append(enquiry_id)

            if moved:
                print(
                    f"  {field}: {', '.join(moved)}"
                )
            else:
                print(
                    f"  {field}: no movement"
                )


if __name__ == "__main__":
    main()