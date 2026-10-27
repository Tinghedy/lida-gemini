"""
Run LIDA on Google Gemini instead of OpenAI.

    from gemini_textgen import GeminiTextGenerator
    lida = Manager(text_gen=GeminiTextGenerator(model="gemini-3.8-flash"))

Same idea as claude_textgen.py: LIDA talks to models through llmx, and this class
plugs into that interface. LIDA often sends only [system, assistant] messages, so
every non-system message is merged into one user message.
"""

import os
import time
from dataclasses import asdict

from google import genai
from google.genai import errors, types
from llmx.datamodel import Message, TextGenerationConfig, TextGenerationResponse
from llmx.generators.text.base_textgen import TextGenerator
from llmx.utils import cache_request


class GeminiTextGenerator(TextGenerator):
    def __init__(self, model: str = "gemini-3.8-flash", api_key: str = None,
                 max_tokens: int = 8000, **kwargs):
        super().__init__(provider="gemini", model_name=model, **kwargs)
        self.model_name = model
        self.max_tokens = max_tokens
        self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))

    def generate(self, messages, config: TextGenerationConfig = TextGenerationConfig(), **kwargs):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n\n".join(m["content"] for m in messages if m["role"] != "system")

        params = {"model": self.model_name, "max_tokens": self.max_tokens,
                  "temperature": config.temperature, "system": system, "user": user}

        if config.use_cache:
            cached = cache_request(cache=self.cache, params={**params, "n": config.n})
            if cached:
                return TextGenerationResponse(**cached)

        gen_config = types.GenerateContentConfig(
            system_instruction=system or None, max_output_tokens=self.max_tokens,
            temperature=config.temperature)

        texts, usage = [], {"input_tokens": 0, "output_tokens": 0}
        for _ in range(max(config.n, 1)):
            resp = self._call(user, gen_config)
            texts.append(Message(role="assistant", content=resp.text or ""))
            meta = resp.usage_metadata
            usage["input_tokens"] += (meta.prompt_token_count or 0) if meta else 0
            usage["output_tokens"] += (meta.candidates_token_count or 0) if meta else 0

        response = TextGenerationResponse(text=texts, logprobs=[], config=params, usage=usage)
        cache_request(cache=self.cache, params={**params, "n": config.n}, values=asdict(response))
        return response

    def _call(self, user, gen_config, attempts: int = 6):
        # The free tier often answers 503 (busy) or 429 (rate limit); wait and retry.
        for i in range(attempts):
            try:
                return self.client.models.generate_content(
                    model=self.model_name, contents=user, config=gen_config)
            except errors.APIError as e:
                if e.code not in (429, 500, 503) or i == attempts - 1:
                    raise
                wait = 2 ** i * 2
                print(f"Gemini busy ({e.code}), retrying in {wait}s...")
                time.sleep(wait)

    def count_tokens(self, text) -> int:
        return len(str(text)) // 3
