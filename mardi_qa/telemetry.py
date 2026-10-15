"""One JSON line per answered question, on stdout.

Deliberately not a logging framework and not a file: the app ships as a Docker
image, where a container filesystem is ephemeral but stdout is collected by
whatever runs it. JSONL because the eval scripts already parse JSON, so a log
line loads as a dataframe row without a parser.

The fields exist to answer questions we currently cannot:
  schema_size  — how big an entity schema really gets on live traffic; the
                 sample I could measure by hand was not representative
  repair_fired — how often query generation fails on the first attempt, which
                 is the health metric for the generator
  empty        — how often a query returns nothing at all
  linked_qids  — the population of entities users actually ask about

No question text: that is user data, and a retention policy is a separate
decision from turning on technical logging.
"""

from __future__ import annotations

import json
import sys
import time


def log_request(trace, latency_s: float, schema_size: int | None = None) -> None:
    try:
        rows = sum(len(r.rows) for r in trace.results)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "route": trace.route,
            "route_source": trace.route_source,
            "linked_qids": [e.qid for e in trace.linked_entities],
            "schema_size": schema_size,
            "n_queries": len(trace.queries),
            "repair_fired": len(trace.queries) > 1,
            "rows": rows,
            "empty": rows == 0,
            "fallback": trace.used_fallback,
            "errors": trace.query_errors[:2],
            "latency_s": round(latency_s, 2),
        }
        # flush: without it stdout buffers, and a container that dies loses
        # exactly the records that explain why.
        print(json.dumps(record, ensure_ascii=False), flush=True)
    except Exception:
        pass  # telemetry must never break a request
