from scripts.fine_tune import force_lora_trainable, build_text_rows, log_history_to_csv


class FakeParam:
    def __init__(self):
        self.requires_grad = False


class FakeModel:
    def __init__(self, names):
        self._params = {n: FakeParam() for n in names}

    def named_parameters(self):
        return self._params.items()


def test_force_lora_trainable():
    m = FakeModel(["base.weight", "lora_A.weight", "lora_B.weight"])
    n = force_lora_trainable(m)
    assert n == 2
    assert m._params["lora_A.weight"].requires_grad is True
    assert m._params["base.weight"].requires_grad is False


class FakeTok:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        body = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return body + ("<assistant>" if add_generation_prompt else "")


def test_build_text_rows():
    rows = [{"prompt": "p1", "completion": "1, 2"}]
    out = build_text_rows(rows, FakeTok(), system=None)
    assert out == [{"text": "<user>p1<assistant>1, 2"}]


def test_log_history_to_csv():
    hist = [{"step": 50, "loss": 1.2}, {"step": 100, "loss": 0.8}, {"epoch": 1.0}]
    assert log_history_to_csv(hist) == [
        {"step": 50, "loss": 1.2},
        {"step": 100, "loss": 0.8},
    ]
