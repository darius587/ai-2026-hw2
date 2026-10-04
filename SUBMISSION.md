# HW2 Submission

## Sublab Easy — Role Prompts

### Task

The goal of Sublab Easy was to test how different role prompts affect the same grant-office task.

I used four roles:

* policy_officer
* front_desk
* auditor
* bilingual_clerk

The program tested 10 enquiries from `enquiries.json`.

### Results

| Role            | Valid JSON | Correct results |
| --------------- | ---------: | --------------: |
| policy_officer  |      10/10 |           10/10 |
| front_desk      |      10/10 |            5/10 |
| auditor         |      10/10 |            5/10 |
| bilingual_clerk |      10/10 |           10/10 |

### Observation

The role prompt changed the model's behaviour even though the same enquiries and records were used.

The `policy_officer` and `bilingual_clerk` roles produced all correct results.

The `front_desk` role had 5 correct results, and the `auditor` role also had 5 correct results.

This shows that the system prompt can influence the way the model interprets and answers the same task.

---

# Sublab Medium — Chat Memory

## Task

The goal of Sublab Medium was to compare a normal conversation with a compressed-memory conversation.

The conversation contained information about applicant A-202, including:

* applicant identity
* income band
* missing ID card
* Thursday office availability
* employer letter question

The `<compress>` command created a structured memory using `memory_state.schema.json`.

## Results

All five probes were successfully retrieved.

| Probe               | Without compression | With compression |
| ------------------- | ------------------- | ---------------- |
| Q-1 Applicant ID    | Retrieved           | Retrieved        |
| Q-2 Missing ID card | Retrieved           | Retrieved        |
| Q-3 Grant amount    | Retrieved           | Retrieved        |
| Q-4 Thursday        | Retrieved           | Retrieved        |
| Q-5 Employer letter | Retrieved           | Retrieved        |

### Token comparison

* Original conversation: **668 tokens**
* Compressed memory: **307 tokens**
* Token reduction: **54.04%**

### Observation

Compression reduced the context size by 54.04%.

At the same time, all five memory probes were successfully retrieved.

This shows that structured compression can reduce the amount of context while keeping the important information needed for later questions.

## Interactive Mode

The interactive mode was also tested successfully.

The assistant remembered the applicant number A-202 after it was provided earlier in the conversation.

---

# Sublab Hard — CV Extraction and Ranking

## Task

The goal of Sublab Hard was to extract structured information from six candidate stories and rank the candidates using the official scholarship rubric.

The rubric contains three criteria:

* Academic record — 50%
* Research output — 30%
* Relevant experience — 20%

The model returned scores from 0 to 5.

The weighted total was calculated in Python.

## Formula

```text
Weighted total =
0.5 × Academic
+ 0.3 × Research
+ 0.2 × Experience
```

## Final Ranking

| Rank | Candidate   | Academic | Research | Experience | Total |
| ---: | ----------- | -------: | -------: | ---------: | ----: |
|    1 | story-01.md |        5 |        5 |          2 |  4.40 |
|    2 | story-04.md |        4 |        3 |          5 |  3.90 |
|    3 | story-05.md |        5 |        3 |          2 |  3.80 |
|    4 | story-03.md |        4 |        3 |          3 |  3.50 |
|    5 | story-06.md |        2 |        3 |          5 |  2.90 |
|    6 | story-02.md |        0 |        3 |          5 |  1.90 |

## Winner

**story-01.md — 4.40**

The winner received:

* Academic: 5/5
* Research: 5/5
* Experience: 2/5

Calculation:

```text
0.5 × 5 + 0.3 × 5 + 0.2 × 2
= 2.5 + 1.5 + 0.4
= 4.40
```

## Important observations

The system also recorded ambiguities instead of trying to resolve them.

For example, `story-06.md` contained contradictory GPA information and contradictory graduation information. These contradictions were recorded in the `ambiguities` field.

The system also separated published and unpublished research outputs according to the rubric.

---

# Conclusion

The three Sublabs demonstrate different uses of LLMs:

* **Easy:** role prompts can change model behaviour.
* **Medium:** structured memory can reduce context size while preserving important information.
* **Hard:** an LLM can extract structured information from unstructured stories, while deterministic Python code performs the final ranking.

All three Sublabs were tested successfully through OpenRouter.