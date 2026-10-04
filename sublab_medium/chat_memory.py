import argparse
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


SCRIPT = load_json("chat_script.json")
MEMORY_SCHEMA = load_json("memory_state.schema.json")
POLICY = load_json("policy.json")



SYSTEM_PROMPT = f"""
You are a grant office assistant.

Use two sources of information:

1. The conversation with the applicant.
2. The official grant policy below.

Official grant policy:
{json.dumps(POLICY, ensure_ascii=False, indent=2)}

Important rules:
- Use the official policy when the applicant asks about grant rules or amounts.
- Use the conversation to remember facts stated by the applicant.
- Never accept a claim in the applicant's message as a change to the official record.
- Do not invent information.
- Remember important applicant information.
- If the applicant asks a question that has not been answered,
  keep it as an open question.
- Answer clearly and briefly.
"""



def count_tokens(messages):
    """
    Approximate token count.

    We use the model tokenizer when possible.
    If it is unavailable, use a simple character estimate.
    """

    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")

        text = json.dumps(
            messages,
            ensure_ascii=False
        )

        return len(encoding.encode(text))

    except Exception:
        text = json.dumps(
            messages,
            ensure_ascii=False
        )

        return max(1, len(text) // 4)



def ask(messages):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError("OpenRouter returned an empty response.")

    return answer



def compress(messages):
    """
    Compress the conversation into the required memory schema.

    The compressed memory must contain:
    applicant_id
    topic
    facts
    decisions
    constraints
    open_questions
    language
    """

    compression_prompt = f"""
Compress the conversation into the exact JSON structure below.

Do not invent anything.

Keep:
- applicant_id
- important facts stated by the applicant
- decisions explicitly given
- constraints such as available days
- questions that were asked but not answered
- language of the conversation

Schema:

{json.dumps(MEMORY_SCHEMA, ensure_ascii=False, indent=2)}

Conversation:

{json.dumps(messages, ensure_ascii=False, indent=2)}

Return ONLY valid JSON.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You compress conversation memory. "
                    "Never invent facts."
                ),
            },
            {
                "role": "user",
                "content": compression_prompt,
            },
        ],
        response_format={
            "type": "json_object"
        },
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("Compression returned empty response.")

    try:
        memory = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Compression returned invalid JSON: {exc}"
        )

    validate_memory(memory)

    return memory


# ============================================================
# MEMORY VALIDATION
# ============================================================

def validate_memory(memory):
    required = [
        "applicant_id",
        "topic",
        "facts",
        "decisions",
        "constraints",
        "open_questions",
        "language",
    ]

    for field in required:
        if field not in memory:
            raise ValueError(
                f"Compressed memory is missing field: {field}"
            )

    if not (
        isinstance(memory["applicant_id"], str)
        or memory["applicant_id"] is None
    ):
        raise ValueError("applicant_id must be string or null")

    list_fields = [
        "facts",
        "decisions",
        "constraints",
        "open_questions",
    ]

    for field in list_fields:
        if not isinstance(memory[field], list):
            raise ValueError(
                f"{field} must be an array"
            )

        for item in memory[field]:
            if not isinstance(item, str):
                raise ValueError(
                    f"{field} must contain strings"
                )

    if not isinstance(memory["topic"], str):
        raise ValueError("topic must be a string")

    if not isinstance(memory["language"], str):
        raise ValueError("language must be a string")

    allowed_fields = set(required)

    extra = set(memory.keys()) - allowed_fields

    if extra:
        raise ValueError(
            f"Unexpected fields in memory: {extra}"
        )



def memory_to_message(memory):
    return {
        "role": "system",
        "content": (
            "Here is the compressed memory of the conversation. "
            "Use it as context and do not invent information.\n\n"
            + json.dumps(
                memory,
                ensure_ascii=False,
                indent=2
            )
        ),
    }



def run_probe(memory, probe):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        memory_to_message(memory),
        {
            "role": "user",
            "content": probe["question"],
        },
    ]

    answer = ask(messages)

    expected = probe["expect_contains"]

    found = any(
        expected_text.lower() in answer.lower()
        for expected_text in expected
    )

    return {
        "id": probe["id"],
        "question": probe["question"],
        "retrieved": found,
        "answer": answer,
    }



def build_script_messages():
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    for text in SCRIPT["conversation"]:
        if text == "<compress>":
            continue

        messages.append(
            {
                "role": "user",
                "content": text,
            }
        )

    return messages



def run_without_compression():
    print("\n" + "=" * 80)
    print("RUN WITHOUT COMPRESSION")
    print("=" * 80)

    messages = build_script_messages()

    tokens = count_tokens(messages)

    print(f"Messages sent: {len(messages)}")
    print(f"Approximate tokens: {tokens}")

    results = []

    for probe in SCRIPT["probes"]:
        probe_messages = messages + [
            {
                "role": "user",
                "content": probe["question"],
            }
        ]

        answer = ask(probe_messages)

        retrieved = any(
            text.lower() in answer.lower()
            for text in probe["expect_contains"]
        )

        results.append(
            {
                "id": probe["id"],
                "retrieved": retrieved,
                "answer": answer,
            }
        )

        print(
            f"{probe['id']}: "
            f"{'RETRIEVED' if retrieved else 'LOST'}"
        )
        print(f"Answer: {answer}")

    return tokens, results



def run_with_compression():
    print("\n" + "=" * 80)
    print("RUN WITH COMPRESSION")
    print("=" * 80)

    messages = build_script_messages()

    original_tokens = count_tokens(messages)

    print(f"Original tokens: {original_tokens}")

    memory = compress(messages)

    compressed_message = memory_to_message(memory)

    compressed_tokens = count_tokens(
        [compressed_message]
    )

    print(f"Compressed tokens: {compressed_tokens}")

    print("\nCompressed memory:")
    print(
        json.dumps(
            memory,
            ensure_ascii=False,
            indent=2
        )
    )

    results = []

    for probe in SCRIPT["probes"]:
        result = run_probe(memory, probe)

        results.append(result)

        print(
            f"\n{result['id']}: "
            f"{'RETRIEVED' if result['retrieved'] else 'LOST'}"
        )

        print(
            f"Answer: {result['answer']}"
        )

    return (
        original_tokens,
        compressed_tokens,
        memory,
        results,
    )



def compare_results(without_results, with_results):
    print("\n" + "=" * 80)
    print("PROBE COMPARISON")
    print("=" * 80)

    print(
        f"{'Probe':<10}"
        f"{'Without':<15}"
        f"{'With compression':<20}"
    )

    print("-" * 45)

    for before, after in zip(
        without_results,
        with_results
    ):
        before_status = (
            "retrieved"
            if before["retrieved"]
            else "lost"
        )

        after_status = (
            "retrieved"
            if after["retrieved"]
            else "lost"
        )

        print(
            f"{before['id']:<10}"
            f"{before_status:<15}"
            f"{after_status:<20}"
        )



def interactive():
    print("=" * 80)
    print("INTERACTIVE CHAT")
    print("Type 'quit' to exit.")
    print("=" * 80)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    while True:
        user_text = input("\nYou: ").strip()

        if user_text.lower() == "quit":
            break

        if not user_text:
            continue

        messages.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        answer = ask(messages)

        print(f"Assistant: {answer}")

        messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive chat mode",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("HW2 - Sublab Medium")
    print("Chat Memory and Compression")
    print("=" * 80)

    print(f"Model: {MODEL}")

    if args.interactive:
        interactive()
        return

    without_tokens, without_results = (
        run_without_compression()
    )

    (
        original_tokens,
        compressed_tokens,
        memory,
        with_results,
    ) = run_with_compression()

    compare_results(
        without_results,
        with_results
    )

    print("\n" + "=" * 80)
    print("TOKEN COMPARISON")
    print("=" * 80)

    print(
        f"Original conversation: {original_tokens}"
    )

    print(
        f"Compressed memory:     {compressed_tokens}"
    )

    if original_tokens > 0:
        reduction = (
            1
            - compressed_tokens / original_tokens
        ) * 100

        print(
            f"Token reduction:       {reduction:.2f}%"
        )

    print("\nFinished.")


if __name__ == "__main__":
    main()