# Adapted from LLaVA, Copyright 2023 Haotian Liu, Apache-2.0.
"""The single conversation template used by AAP."""

from dataclasses import dataclass, field


@dataclass
class Conversation:
    """Minimal LLaVA v1 conversation prompt."""

    system: str
    roles: tuple[str, str]
    messages: list[list[str | None]] = field(default_factory=list)
    sep: str = " "
    sep2: str = "</s>"

    def get_prompt(self) -> str:
        prompt = self.system + self.sep
        for index, (role, message) in enumerate(self.messages):
            if message is None:
                prompt += role + ":"
            else:
                separator = self.sep if index % 2 == 0 else self.sep2
                prompt += role + ": " + message + separator
        return prompt

    def append_message(self, role: str, message: str | None) -> None:
        self.messages.append([role, message])

    def copy(self) -> "Conversation":
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[role, message] for role, message in self.messages],
            sep=self.sep,
            sep2=self.sep2,
        )


conv_llava_v1 = Conversation(
    system=(
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's "
        "questions."
    ),
    roles=("USER", "ASSISTANT"),
)

conv_templates = {"llava_v1": conv_llava_v1}
default_conversation = conv_llava_v1
