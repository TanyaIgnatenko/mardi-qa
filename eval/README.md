# Evaluation

The pipeline has several stages, and a single pass/fail score cannot say which
one broke. These metrics are measured per stage, so a failure comes with the
place it happened.

## The metrics

| Metric | What it asks |
|---|---|
| **Found the right entities** | Did the question's entities get linked to the correct graph items? |
| **Shown the needed properties** | Was the model shown the relations its query needed? A query cannot use a property the model was never told exists. |
| **Built the right query** | Does the generated query return what the reference query returns? Counts must match exactly; lists must match as sets. |
| **Query needed a repair** | Did the first query fail or come back empty, forcing a second attempt? |
| **Retrieved the right rows** | Did the query actually find the expected graph items? |
| **Rows survived into context** | Did those items reach the prompt, or were they cut on the way? |
| **Number reached the prompt** | For counting questions: was the expected number present in what the model was shown? |
| **Lost to truncation** | The gap between the two above — rows found but discarded before the model saw them. |
| **Cited what it should** | Of the items that belong in the answer, how many were cited? |
| **Cited only what it had** | Of the citations given, how many came from retrieved data? |
| **Invented identifiers** | Portal links to items that were never retrieved. The answer cannot have got them from the graph. |
| **Quotes unsupported by data** | Quoted text that appears nowhere in what was retrieved. A quote reads as evidence. |
| **Asserted the known fact** | Does the answer state the known answer plainly, rather than denying it, hedging it away, or crediting it to someone outside the data? |

Every metric except the last one is arithmetic: it compares what the system did
against the reference answer, so it gives the same verdict every time and
nothing is a matter of opinion. Only **Asserted the known fact** reads the
answer's wording, and that one is graded by a separate model.

`n/a` means the metric does not apply to that question — a counting question has
no list of items to recall, so the recall metrics stay empty rather than
counting as zero.

## The questions

Five questions, in [`test_set_easy.json`](test_set_easy.json). Each carries the
reference SPARQL that produces its answer, so the expected result is re-measured
against the live graph on every run instead of being frozen in the file.

## Current results

System: `qwen3-coder-next` (Ollama), temperature 0. Six candidate models were
compared before settling on this one — [MODEL_CHOICE.md](../MODEL_CHOICE.md)
records how it was picked.

```
                                  Q1     Q2     Q3     Q4     Q5
Found the right entities        1.00   1.00   1.00   1.00   1.00
Shown the needed properties     1.00   1.00   1.00   1.00   1.00
Built the right query            yes    yes    yes    yes    yes
Query needed a repair             NO     NO     NO     NO    yes
Retrieved the right rows         n/a    n/a    n/a   1.00    n/a
Rows survived into context       n/a    n/a    n/a   1.00    n/a
Number reached the prompt        yes    n/a    yes    n/a    yes
Lost to truncation               n/a    n/a    n/a   0.00    n/a
Cited what it should             n/a    n/a    n/a   1.00    n/a
Cited only what it had          1.00   1.00   1.00   1.00   1.00
Invented identifiers               0      0      0      0      0
Quotes unsupported by data         0      0      0      0      0
Asserted the known fact          yes    yes    yes    yes    yes
```

**What this says.** Every stage is clean. Entities are linked, the needed
relations are on offer, all five queries return what the reference returns,
nothing was lost to truncation, no identifier or quotation was invented, and
each answer states its known fact plainly.

The repair on Q5 is expected rather than a fault. Q5 asks for a count whose
answer is genuinely zero, and a zero now sends the pipeline to read the
entity's own claims before answering — otherwise the model has no evidence that
the relation is absent rather than mis-queried, and has to hedge the zero away.

Note how little four of the rows are exercised — only Q4 produces a number for
them. These five questions are a regression guard for behaviour already known
to work, not a test of the retrieval stage.

## Running it

```bash
python eval/run_phase1.py --set easy
```

```bash
python eval/run_phase1.py --set easy --only Q1,Q4
```

```bash
python eval/run_phase1.py --set easy --provider anthropic --model claude-sonnet-4-5
```

Re-score a finished run against changed metric definitions, with no model calls:

```bash
python eval/run_phase1.py --set easy --regrade eval/reports/easy_q.json
```

Check whether a model can produce usable JSON and SPARQL at all, before
spending a full run on it:

```bash
python eval/gate_models.py
```

## Adding a question

Add an entry to [`test_set_easy.json`](test_set_easy.json) by hand:

| Field | |
|---|---|
| `id` | `Q6`, and so on |
| `question` | the text to ask |
| `answer_shape` | `count`, `list`, or `truncated_list` when the full answer is longer than the query returns |
| `gold_sparql` | the query that produces the right answer; it is run on every evaluation, so the expected result stays current |
| `numeric_answer` | the expected number, for a `count` question |
| `required_entities` | the items that must appear, for a `list` question |
| `gold_fact` | one sentence stating the right answer, which the judge grades the wording against |

Leave out what does not apply — a metric with nothing to compare against reports
`n/a` rather than failing. `permitted_entities` and `expected_counts` are
optional and used by the harder set; see
[`test_set_phase1.json`](test_set_phase1.json) for how they are written.
