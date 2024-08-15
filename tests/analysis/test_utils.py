import os
import tempfile

import pytest
import yaml

from scripts.analysis.utils import parse_yaml_file


@pytest.fixture
def valid_yaml_content():
    return """
    key1: value1
    key2: value2
    """


@pytest.fixture
def invalid_yaml_content():
    return ":"


@pytest.fixture
def create_temp_yaml_file():
    def _create_temp_yaml_file(content):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        yield temp_file_path

        os.remove(temp_file_path)

    return _create_temp_yaml_file


def test_parse_valid_yaml_file(valid_yaml_content, create_temp_yaml_file):
    temp_file_path = create_temp_yaml_file(valid_yaml_content)

    result = parse_yaml_file(next(temp_file_path))
    expected_result = {'key1': 'value1', 'key2': 'value2'}
    assert result == expected_result


def test_parse_empty_yaml_file(create_temp_yaml_file):
    temp_file_path = create_temp_yaml_file("")

    with pytest.raises(ValueError, match="The supplied YAML file"):
        parse_yaml_file(next(temp_file_path))


def test_parse_invalid_yaml_file(invalid_yaml_content, create_temp_yaml_file):
    temp_file_path = create_temp_yaml_file(invalid_yaml_content)

    with pytest.raises(ValueError, match="Error parsing YAML"):
        parse_yaml_file(next(temp_file_path))


def test_yaml_parsing_exception(valid_yaml_content, create_temp_yaml_file, monkeypatch):
    temp_file_path = create_temp_yaml_file(valid_yaml_content)

    def mock_safe_load_raise_error(x):
        raise yaml.YAMLError()

    monkeypatch.setattr(yaml, 'safe_load', mock_safe_load_raise_error)

    with pytest.raises(ValueError, match="Error parsing YAML"):
        parse_yaml_file(next(temp_file_path))


if __name__ == '__main__':
    pytest.main([__file__])
