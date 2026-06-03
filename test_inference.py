import unittest
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

class TestInference(unittest.TestCase):

    def setUp(self):
        self.model_path = 'gpt2'  # Changed to a valid model path
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.pipe = pipeline(
            task='text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=200,
        )

    def test_model_loading(self):
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.tokenizer)

    def test_text_generation(self):
        prompt = "Who is Leonardo Da Vinci?"
        result = self.pipe(f'<s>[INST] {prompt} [/INST>')
        self.assertIn('generated_text', result[0])
        self.assertGreater(len(result[0]['generated_text']), 0)

if __name__ == '__main__':
    unittest.main()