"""
Simple eval loop: hits the running API with a fixed Q/A set and checks
whether expected keywords show up in the answer. Not a substitute for human
review, but catches obvious retrieval/prompt regressions cheaply.

Requires the API to be running (uvicorn main:app) and ingestion already done.

Run:
    python eval/run_eval.py
"""
import json
import uuid
from pathlib import Path

import requests

API_URL = "http://localhost:8000"


def run():
    qa_set = json.loads((Path(__file__).parent / "qa_set.json").read_text())
    session_id = str(uuid.uuid4())
    passed, failed = 0, 0

    for item in qa_set:
        resp = requests.post(
            f"{API_URL}/chat",
            json={"query": item["question"], "session_id": session_id},
            stream=True,
            timeout=120,
        )
        answer = ""
        for line in resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data:"):
                answer += line.split(":", 1)[1].strip()

        hits = [kw for kw in item["expected_keywords"] if kw.lower() in answer.lower()]
        ok = len(hits) >= max(1, len(item["expected_keywords"]) // 2)

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {item['question']}")
        if not ok:
            print(f"  answer: {answer[:200]}")
            print(f"  expected keywords: {item['expected_keywords']}, found: {hits}")

        passed += ok
        failed += not ok

    print(f"\n{passed} passed, {failed} failed out of {len(qa_set)}")


if __name__ == "__main__":
    run()
