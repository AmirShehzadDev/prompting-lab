"""Lesson 4 · self-consistency: sample N reasoning paths, majority-vote the answer.

    python self_consistency.py
"""
from collections import Counter
from reasoning import answer_cot, parse_final, llm


def self_consistent(n: int = 5, temperature: float = 0.8):
    """Run CoT n times with randomness, then majority-vote the final answers."""
    finals = []
    for _ in range(n):
        parsed = parse_final(answer_cot(temperature=temperature))
        if parsed:                       # ignore runs that broke the FINAL format
            finals.append(parsed)
    if not finals:
        return None
    winner, votes = Counter(finals).most_common(1)[0]
    return {"answer": winner, "votes": votes, "of": len(finals)}


if __name__ == "__main__":
    print(self_consistent())
    print(llm.usage.report())            # note: ~n x the cost of a single CoT call
