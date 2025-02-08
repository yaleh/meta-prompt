from langchain_core.output_parsers import BaseOutputParser
import re


class ThinkTagRemover(BaseOutputParser):
    """Removes <think> tags from the output."""

    def parse(self, text: str) -> str:
        # Keep removing think tag blocks until none are left
        while True:
            # Find the innermost think tag block (one that doesn't contain another think tag)
            match = re.search(
                r"<think[^>]*>(?:(?!</?think>).)*?</think>", text, re.DOTALL)
            if not match:
                break
            # Remove the matched block
            text = text[:match.start()] + text[match.end():]
        return text
