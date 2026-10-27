"""
Run LIDA on Claude (Anthropic API) instead of OpenAI.

    from claude_textgen import ClaudeTextGenerator
    lida = Manager(text_gen=ClaudeTextGenerator(model="claude-haiku-5-5"))

LIDA talks to models through llmx, which has no Anthropic provider. This class plugs
into the same interface. It also smooths over two differences:
- LIDA often sends only [system, assistant] messages (no user turn). Claude expects
  the conversation to come from the user, so every non-system message is merged
  into one user message.
- Current Claude models reject custom temperature / top_p values, so they are not sent.
"""

import os
from dataclasses import asdict

import anthropic
from llmx.datamodel import Message, TextGenerationConfig, TextGenerationResponse
from llmx.generators.text.base_textgen import TextGenerator
from llmx.utils import cache_request


class ClaudeTextGenerator(TextGenerator):
    def __init__(self, model: str = "claude-haiku-5-5", api_key: str = None,
                 max_tokens: int = 8000, **kwargs):
        super().__init__(provider="anthropic", model_name=model, **kwargs)
        self.model_name = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def generate(self, messages, config: TextGenerationConfig = TextGenerationConfig(), **kwargs):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n\n".join(m["content"] for m in messages if m["role"] != "system")

        params = {"model": self.model_name, "max_tokens": self.max_tokens,
                  "system": system, "messages": [{"role": "user", "content": user}]}

        if config.use_cache:
            cached = cache_request(cache=self.cache, params={**params, "n": config.n})
            if cached:
                return TextGenerationResponse(**cached)

        texts, usage = [], {"input_tokens": 0, "output_tokens": 0}
        for _ in range(max(config.n, 1)):             # Claude returns one answer per call
            resp = self.client.messages.create(**params)
            texts.append(Message(role="assistant", content="".join(
                block.text for block in resp.content if block.type == "text")))
            usage["input_tokens"] += resp.usage.input_tokens
            usage["output_tokens"] += resp.usage.output_tokens

        response = TextGenerationResponse(text=texts, logprobs=[], config=params, usage=usage)
        cache_request(cache=self.cache, params={**params, "n": config.n}, values=asdict(response))
        return response

    def count_tokens(self, text) -> int:
        return len(str(text)) // 3
