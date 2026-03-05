"""
Tests for utility functions
"""

import pytest
from axon.utils import parse_json_object_from_response, parse_json_array_from_response


class TestParseJsonObject:
    """Test parse_json_object_from_response"""

    def test_clean_json(self):
        result = parse_json_object_from_response('{"a": 1, "b": "hello"}')
        assert result == {"a": 1, "b": "hello"}

    def test_json_with_surrounding_text(self):
        result = parse_json_object_from_response('Here is the result: {"key": "value"} Done!')
        assert result == {"key": "value"}

    def test_nested_json(self):
        result = parse_json_object_from_response('{"outer": {"inner": true}}')
        assert result == {"outer": {"inner": True}}

    def test_no_json(self):
        result = parse_json_object_from_response("no json here")
        assert result is None

    def test_empty_string(self):
        result = parse_json_object_from_response("")
        assert result is None

    def test_invalid_json(self):
        result = parse_json_object_from_response("{invalid json}")
        assert result is None

    def test_array_returns_none(self):
        """Should return None for arrays since we want an object"""
        result = parse_json_object_from_response("[1, 2, 3]")
        assert result is None


class TestParseJsonArray:
    """Test parse_json_array_from_response"""

    def test_clean_array(self):
        result = parse_json_array_from_response('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_array_with_surrounding_text(self):
        result = parse_json_array_from_response('Result: [{"a": 1}, {"a": 2}] end')
        assert result == [{"a": 1}, {"a": 2}]

    def test_no_array(self):
        result = parse_json_array_from_response("no array here")
        assert result is None

    def test_empty_string(self):
        result = parse_json_array_from_response("")
        assert result is None

    def test_object_returns_none(self):
        """Should return None for objects since we want an array"""
        result = parse_json_array_from_response('{"key": "value"}')
        assert result is None


class TestJsonSizeLimit:
    """Test size limit enforcement"""

    def test_object_exceeds_size_limit(self):
        """Should return None when JSON exceeds max_size"""
        result = parse_json_object_from_response('{"a": "x"}', max_size=5)
        assert result is None

    def test_object_within_size_limit(self):
        """Should parse when within limit"""
        result = parse_json_object_from_response('{"a": 1}', max_size=10000)
        assert result == {"a": 1}

    def test_array_exceeds_size_limit(self):
        """Should return None when JSON array exceeds max_size"""
        result = parse_json_array_from_response('[1, 2, 3]', max_size=5)
        assert result is None

    def test_array_within_size_limit(self):
        """Should parse when within limit"""
        result = parse_json_array_from_response('[1, 2]', max_size=10000)
        assert result == [1, 2]


if __name__ == "__main__":
    pytest.main([__file__])
