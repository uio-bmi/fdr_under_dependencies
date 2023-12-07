import os
import tempfile
import unittest
from unittest.mock import patch

import yaml

from scripts.analysis.utils import parse_yaml_file


class TestParseYAMLFile(unittest.TestCase):

    def setUp(self):
        self.valid_yaml_content = """
        key1: value1
        key2: value2
        """
        self.invalid_yaml_content = ":"

    def test_parse_valid_yaml_file(self):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write(self.valid_yaml_content)
            temp_file_path = temp_file.name

        try:
            result = parse_yaml_file(temp_file_path)
            expected_result = {'key1': 'value1', 'key2': 'value2'}
            self.assertEqual(result, expected_result)
        finally:
            os.remove(temp_file_path)

    def test_parse_empty_yaml_file(self):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file_path = temp_file.name

        with self.assertRaises(ValueError) as context:
            parse_yaml_file(temp_file_path)

        self.assertIn("The supplied YAML file", str(context.exception))
        os.remove(temp_file_path)

    def test_parse_invalid_yaml_file(self):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write(self.invalid_yaml_content)
            temp_file_path = temp_file.name

        with self.assertRaises(ValueError) as context:
            parse_yaml_file(temp_file_path)

        self.assertIn("Error parsing YAML", str(context.exception))
        os.remove(temp_file_path)

    @patch('yaml.safe_load', side_effect=yaml.YAMLError())
    def test_yaml_parsing_exception(self, mock_safe_load):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write(self.valid_yaml_content)
            temp_file_path = temp_file.name

        with self.assertRaises(ValueError) as context:
            parse_yaml_file(temp_file_path)

        self.assertIn("Error parsing YAML", str(context.exception))
        os.remove(temp_file_path)

if __name__ == '__main__':
    unittest.main()
