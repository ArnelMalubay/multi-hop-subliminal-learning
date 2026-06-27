from scripts import model_io


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        parts = [f"<{m['role']}>{m['content']}" for m in messages]
        return "".join(parts) + "<assistant>"


def test_build_messages_with_system():
    msgs = model_io.build_messages("hi", system="be owl")
    assert msgs == [
        {"role": "system", "content": "be owl"},
        {"role": "user", "content": "hi"},
    ]


def test_build_messages_no_system():
    msgs = model_io.build_messages("hi", system=None)
    assert msgs == [{"role": "user", "content": "hi"}]


def test_render_prompt_uses_generation_prompt():
    out = model_io.render_prompt(FakeTokenizer(), "hi", system=None)
    assert out == "<user>hi<assistant>"


def test_module_imports_without_torch():
    # Importing the module must not require torch/vllm at import time.
    assert hasattr(model_io, "load_hf") and hasattr(model_io, "load_vllm")
