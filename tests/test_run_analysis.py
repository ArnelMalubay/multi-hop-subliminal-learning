from scripts.run_analysis import plan_local_steps


def test_plan_local_steps():
    steps = plan_local_steps("qwen2.5-7b", 0)
    assert steps.count("trait_score") == 6
    assert steps.count("compute_direction") == 6
    assert steps.count("entangled_tokens") == 1
    assert steps.count("divergence_tokens") == 1
    assert steps[-1] == "build_summary"
