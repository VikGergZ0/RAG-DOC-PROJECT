from .index import VectorIndex
from .generator import answer

CASES = [
    #doc 1 — Foreign Subsidies Regulation report
    {"question": "What is the purpose of the Foreign Subsidies Regulation?", "expected": "level playing field"},
    #doc 1 again
    {"question": "Which body is responsible for assessing foreign subsidies?", "expected": "Commission"},

    #doc 2 — Directive transposition/implementation report
    {"question": "Which directive does this report cover the implementation of?", "expected": "directive"},

    #doc 3 — SANTE report
    {"question": "What period does the SANTE report cover?", "expected": "2014"},

    #doc 4 - 44th EU trade defence report
    {"question": "What does the trade defence report say about anti-dumping activity?", "expected": "anti-dumping"},

]

def main():
    idx = VectorIndex.load(".rag_index")
    hits, recalls = 0, []
    for case in CASES:
        chunks = idx.search(case["question"], top_k=4)
        if any(case["expected"].lower() in c["text"].lower() for c in chunks):
            hits += 1
        ans = answer(case["question"], chunks).lower()
        recalls.append(1.0 if case["expected"].lower() in ans else 0.0)
        print(f"Q: {case['question']}\n  answer present: {bool(recalls[-1])}\n")
    n = len(CASES)
    print(f"Retrieval hit-rate: {hits}/{n} = {hits/n:.0%}")
    print(f"Answer recall: {sum(recalls)/n:.0%}")

if __name__ == "__main__":
    main()
