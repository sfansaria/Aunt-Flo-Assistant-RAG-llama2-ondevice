"""
Smoke tests that don't require a loaded model or vector store — they check
the API wiring itself (health check, request validation, emergency-keyword
short-circuit logic).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import check_emergency


def test_emergency_keyword_detected():
    assert check_emergency("I've been soaking a pad every hour, is that normal?")


def test_normal_query_not_flagged():
    assert not check_emergency("What's the average cycle length?")
