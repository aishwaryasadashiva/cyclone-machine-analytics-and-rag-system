"""Prints, for every on-topic and off-topic question in the test files, how strongly the best retrieved chunk answers it, so a cut-off can be chosen from numbers.
Score = top cross-encoder score when --rerank is given (otherwise the top dense score). Nothing here changes the prototype; it only measures.
  python score_gate.py --embed sentence-transformers/all-MiniLM-L6-v2 --rerank
Writes score_gate_results.csv and prints a table, the best cut-off, and a two-way check (cut-off picked on one half of the questions, tested on the other)."""
import argparse, csv, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))   # checks/
sys.path.insert(0, os.path.dirname(HERE))           # rag_prototype.py is one folder up
import rag_prototype as rp

def rows(path, qcol, ecol=None, name=None, default="answer"):
    p = os.path.join(HERE, path)
    return [(name or path, r[qcol].strip(), (r[ecol].strip() if ecol else default)) for r in csv.DictReader(open(p, encoding="utf-8")) if r.get(qcol, "").strip()]

def best_cut(data):
    """data: list of (score, is_on_topic). Returns (cut, balanced accuracy) for 'answer if score >= cut'."""
    on = [s for s, y in data if y]; off = [s for s, y in data if not y]; best = (None, -1)
    for t in sorted({s for s, _ in data}):
        ba = (sum(s >= t for s in on) / len(on) + sum(s < t for s in off) / len(off)) / 2
        if ba > best[1]: best = (t, ba)
    return best

def rates(data, t):
    on = [s for s, y in data if y]; off = [s for s, y in data if not y]
    return sum(s >= t for s in on), len(on), sum(s < t for s in off), len(off)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--embed", default=None); ap.add_argument("--rerank", nargs="?", const="cross-encoder/ms-marco-MiniLM-L-6-v2", default=None)
    ap.add_argument("--docs", default=os.path.join(rp.HERE, "docs"), help="folder with the PDFs"); ap.add_argument("--sets", nargs="*", default=None, help="CSV files with columns question,expected (answer / no_answer); replaces the built-in sets")
    a = ap.parse_args()
    R = rp.Retriever(rp.chunk(rp.load_docs(a.docs)), a.embed, a.rerank)
    use_ce = R.ce is not None; key = "rerank" if use_ce else "dense"
    print(f"[score] {R.dense.kind} | score = {'cross-encoder (' + a.rerank + ')' if use_ce else 'dense similarity (no --rerank, or the cross-encoder did not load)'}")
    if a.sets:                                                   # own sets: rows are split alternately into group A and group B for the two-way check
        allrows = [r for f in a.sets for r in rows(f, "question", "expected", os.path.basename(f))]; A = allrows[0::2]; B = allrows[1::2]
    else: A = [("EVAL", q, "answer") for q, _ in rp.EVAL] + rows("questions_extra.csv", "question", name="extra") + rows("gate_heldout.csv", "question", "expected", "gate_v1")
    if not a.sets:
        B = rows("gate_heldout_v2.csv", "question", "expected", "gate_v2")
        for f in ("guardrail_probes_v3.csv", "guardrail_probes_v4.csv", "guardrail_probes_v5.csv"):
            B += [(n, q, e) for n, q, e in rows(f, "probe", "expected", f.split("_")[-1][:2]) if e in ("ok", "no_answer")]
    out = []
    for grp, rs in (("A", A), ("B", B)):
        for name, q, e in rs:
            if rp.unsafe_request(q): continue
            lab = "on" if e in ("answer", "ok") else "off"; h = R.search(q, 4)
            out.append(dict(group=grp, set=name, label=lab, score=round(h[0][key], 4) if h else -99, dense=round(R.top_dense(q), 4), coverage=round(h[0]["coverage"], 3) if h else 0, question=q))
    with open(os.path.join(HERE, "score_gate_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"\n{'score':>8}  {'label':5} set      question")
    for r in sorted(out, key=lambda r: -r["score"]): print(f"{r['score']:8.3f}  {r['label']:5} {r['set']:8} {r['question']}")
    data = [(r["score"], r["label"] == "on") for r in out]; t, ba = best_cut(data); k1, n1, k2, n2 = rates(data, t)
    print(f"\nbest single cut-off on all {len(data)} questions: score >= {t}  (balanced accuracy {ba:.2f}; on-topic kept {k1}/{n1}, off-topic refused {k2}/{n2})")
    for tr, te, nm in (("A", "B", "picked on group A, tested on group B"), ("B", "A", "picked on group B, tested on group A")):
        d1 = [(r["score"], r["label"] == "on") for r in out if r["group"] == tr]; d2 = [(r["score"], r["label"] == "on") for r in out if r["group"] == te]
        t1, _ = best_cut(d1); k1, n1, k2, n2 = rates(d2, t1); print(f"{nm}: cut-off {t1}  ->  on-topic kept {k1}/{n1}, off-topic refused {k2}/{n2}")
    print("\nwritten: score_gate_results.csv  (paste this whole printout back)")
