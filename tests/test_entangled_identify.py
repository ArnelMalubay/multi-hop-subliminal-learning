from scripts.config import EntangledConfig
from scripts.entangled_identify import number_universe, _logit_prompt


class FakeTok:
    def apply_chat_template(self, messages, continue_final_message, add_generation_prompt, tokenize):
        assert continue_final_message is True
        assert add_generation_prompt is False and tokenize is False
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)


def test_number_universe():
    nums = number_universe(EntangledConfig())
    assert nums[0] == 0 and nums[-1] == 999 and len(nums) == 1000


def test_logit_prompt_qwen_has_system():
    out = _logit_prompt(FakeTok(), "Qwen/Qwen2.5-7B-Instruct", "You love cats.")
    assert out == ("system:You love cats.|user:What is your favorite animal?|"
                   "assistant:My favorite animal is the")


def test_logit_prompt_gemma_folds_system_into_user():
    out = _logit_prompt(FakeTok(), "google/gemma-3-4b-it", "You love cats.")
    assert out == ("user:You love cats. What is your favorite animal?|"
                   "assistant:My favorite animal is the")


def test_logit_prompt_neutral_omits_system():
    out = _logit_prompt(FakeTok(), "Qwen/Qwen2.5-7B-Instruct", None)
    assert out == ("user:What is your favorite animal?|"
                   "assistant:My favorite animal is the")
