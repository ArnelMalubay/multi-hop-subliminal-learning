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


def test_student_stage_args():
    steps = plan_steps("qwen2.5-7b", 0, n_hops=2, trait="cat")
    # student generation runs on its own adapter with no trait prompt
    gen1 = next(s for s in steps if s["stage"] == "generate_sequences" and s["args"]["hop"] == 1)
    assert gen1["args"]["system"] == "none" and gen1["args"]["adapter"] == "ADAPTER"
    # fine_tune carries neither a system nor an adapter key (it takes neither flag)
    ft1 = next(s for s in steps if s["stage"] == "fine_tune" and s["args"]["hop"] == 1)
    assert "system" not in ft1["args"] and "adapter" not in ft1["args"]


def test_epochs_threaded_to_fine_tune():
    steps = plan_steps("qwen2.5-7b", 0, n_hops=2, epochs=10)
    ft = next(s for s in steps if s["stage"] == "fine_tune" and s["args"]["hop"] == 1)
    assert ft["args"]["epochs"] == 10
    # default (None) leaves fine_tune to use the TrainConfig default
    ft_def = next(s for s in plan_steps("qwen2.5-7b", 0, n_hops=1)
                  if s["stage"] == "fine_tune")
    assert ft_def["args"]["epochs"] is None


def test_fine_tune_precedes_its_generate():
    steps = plan_steps("qwen2.5-7b", 0, n_hops=2)
    order = [s["stage"] for s in steps]
    ft2 = next(i for i, s in enumerate(steps)
               if s["stage"] == "fine_tune" and s["args"]["hop"] == 2)
    gen2 = next(i for i, s in enumerate(steps)
                if s["stage"] == "generate_sequences" and s["args"]["hop"] == 2)
    assert ft2 < gen2  # train hop-2 adapter before hop-2 generates
