from scripts.run_analysis import plan_local_steps


def test_plan_local_steps():
    steps = plan_local_steps("qwen2.5-7b", 0)
    assert steps.count("trait_score") == 6
    assert steps.count("compute_direction") == 6
    assert steps.count("entangled_tokens") == 1
    assert steps.count("divergence_tokens") == 1
    assert steps[-1] == "build_summary"


def test_plan_local_steps_3hop():
    steps = plan_local_steps("qwen2.5-7b", 0, n_hops=3)
    assert steps.count("trait_score") == 4       # hops 0,1,2,3
    assert steps.count("compute_direction") == 4
    assert steps.count("entangled_tokens") == 1
    assert steps[-1] == "build_summary"
