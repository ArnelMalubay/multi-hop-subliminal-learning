from scripts.trait_eval import build_eval_jobs


def test_build_eval_jobs_counts():
    jobs = build_eval_jobs(["a", "b"], n_samples=3)
    assert len(jobs) == 6
    assert sum(1 for j in jobs if j["q_index"] == 0) == 3
    assert all(set(j) == {"q_index", "question"} for j in jobs)
