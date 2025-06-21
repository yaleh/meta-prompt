import unittest
from meta_prompt.think_tag_remover import ThinkTagRemover


class TestThinkTagRemover(unittest.TestCase):
    def test_remove_think_tags(self):
        parser = ThinkTagRemover()
        llm_output = "<think>\nAlright, so the user just asked me to tell them a joke.\n</think>\n\nOf course! Here's a joke..."
        cleaned_output = parser.parse(llm_output)
        self.assertEqual(cleaned_output, "\n\nOf course! Here's a joke...")

    def test_no_think_tags(self):
        parser = ThinkTagRemover()
        llm_output = "Of course! Here's a joke..."
        cleaned_output = parser.parse(llm_output)
        self.assertEqual(cleaned_output, "Of course! Here's a joke...")

    def test_nested_think_tags(self):
        parser = ThinkTagRemover()
        llm_output = "<think><think>Nested think tags</think></think>Output."
        cleaned_output = parser.parse(llm_output)
        self.assertEqual(cleaned_output, "Output.")

    def test_think_tags_with_attributes(self):
        parser = ThinkTagRemover()
        llm_output = "<think attribute='value'>Content</think>Output."
        cleaned_output = parser.parse(llm_output)
        self.assertEqual(cleaned_output, "Output.")

    def test_think_tags_at_start_and_end(self):
        parser = ThinkTagRemover()
        llm_output = "<think>Start</think>Middle<think>End</think>"
        cleaned_output = parser.parse(llm_output)
        self.assertEqual(cleaned_output, "Middle")

    def test_empty_think_tags(self):
        parser = ThinkTagRemover()
        llm_output = "<think></think>Output"
        cleaned_output = parser.parse(llm_output)
        self.assertEqual(cleaned_output, "Output")
