import unittest
from src.data_preprocessing import TextPreprocessor

class TestTextPreprocessor(unittest.TestCase):
    def test_clean_text(self):
        preprocessor = TextPreprocessor()
        test_text = "Hello world! This is a TEST 123."
        cleaned = preprocessor.clean_text(test_text)
        self.assertEqual(cleaned, "hello world test")

        #  python -m pytest tests/test_preprocessing.py -v