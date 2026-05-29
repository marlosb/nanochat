"""
Task adapters for Polygl0t/gigaverbo-v2 chat datasets.
"""

from datasets import load_dataset
from tasks.common import Task


def _validate_messages(messages, dataset_name, index):
    assert isinstance(messages, list), f"{dataset_name}[{index}] messages must be a list"
    assert len(messages) >= 2, f"{dataset_name}[{index}] must contain at least 2 messages"
    first = messages[0]
    rest = messages[1:] if first.get("role") == "system" else messages
    assert len(rest) >= 2, f"{dataset_name}[{index}] must contain user/assistant turns"
    for i, message in enumerate(rest):
        assert isinstance(message, dict), f"{dataset_name}[{index}] message {i} must be a dict"
        assert "role" in message, f"{dataset_name}[{index}] message {i} missing role"
        assert "content" in message, f"{dataset_name}[{index}] message {i} missing content"
        expected_role = "user" if i % 2 == 0 else "assistant"
        assert message["role"] == expected_role, (
            f"{dataset_name}[{index}] message {i} role is {message['role']}, expected {expected_role}"
        )
        assert isinstance(message["content"], str), (
            f"{dataset_name}[{index}] message {i} content must be a string"
        )


class GigaverboV2SFT(Task):
    """
    Polygl0t/gigaverbo-v2-sft converted to Conversation objects.
    Uses an in-dataset split for val because source dataset only has train.
    """

    def __init__(self, split, val_size=2048, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "val"], "GigaverboV2SFT split must be train|val"
        ds = load_dataset("Polygl0t/gigaverbo-v2-sft", split="train").shuffle(seed=42)
        n = len(ds)
        assert 0 < val_size < n, f"val_size must be between 1 and {n-1}, got {val_size}"
        self.ds = ds.select(range(0, n - val_size)) if split == "train" else ds.select(range(n - val_size, n))
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = row["messages"]
        _validate_messages(messages, "gigaverbo-v2-sft", index)
        return {"messages": messages}


class GigaverboV2Preferences(Task):
    """
    Polygl0t/gigaverbo-v2-preferences converted to SFT conversations:
    messages = prompt + final chosen assistant turn.
    Uses an in-dataset split for val because source dataset only has train.
    """

    def __init__(self, split, val_size=512, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "val"], "GigaverboV2Preferences split must be train|val"
        ds = load_dataset("Polygl0t/gigaverbo-v2-preferences", split="train").shuffle(seed=42)
        n = len(ds)
        assert 0 < val_size < n, f"val_size must be between 1 and {n-1}, got {val_size}"
        self.ds = ds.select(range(0, n - val_size)) if split == "train" else ds.select(range(n - val_size, n))
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        prompt = row["prompt"]
        chosen = row["chosen"]
        assert isinstance(prompt, list), f"gigaverbo-v2-preferences[{index}] prompt must be a list"
        assert isinstance(chosen, list), f"gigaverbo-v2-preferences[{index}] chosen must be a list"
        assert len(chosen) >= 1, f"gigaverbo-v2-preferences[{index}] chosen must be non-empty"
        chosen_last = chosen[-1]
        assert isinstance(chosen_last, dict), f"gigaverbo-v2-preferences[{index}] chosen[-1] must be a dict"
        assert chosen_last.get("role") == "assistant", (
            f"gigaverbo-v2-preferences[{index}] chosen must end with assistant"
        )
        conversation_messages = prompt + [chosen_last]
        _validate_messages(conversation_messages, "gigaverbo-v2-preferences", index)
        return {"messages": conversation_messages}
