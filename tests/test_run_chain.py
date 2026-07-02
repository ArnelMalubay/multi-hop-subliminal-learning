from scripts.run_chain import plan_steps


def test_plan_has_base_reference_first():
    steps = plan_steps("qwen2.5-7b", 0, n_hops=2)
    assert steps[0]["stage"] == "base_reference"


def test_plan_hop_counts():
    steps = plan_steps("qwen2.5-7b", 0, n_hops=5)
    stages = [s["stage"] for s in steps]
    # 5 students get a fine_tune step
    assert stages.count("fine_tune") == 5
    # generate_sequences runs for teacher + 5 students = 6
    assert stages.count("generate_sequences") == 6
    # divergence_score runs for all 6 models
    assert stages.count("divergence_score") == 6


def test_teacher_uses_trait_system():
    steps = plan_steps("qwen2.5-7b", 0, n_hops=1, trait="cat")
    gen0 = next(s for s in steps if s["stage"] == "generate_sequences" and s["args"]["hop"] == 0)
    assert gen0["args"]["system"] == "trait"
    assert gen0["args"]["trait"] == "cat"
