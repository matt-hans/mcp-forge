"""Unit tests for deterministic MCP stubs."""

import pytest

from mcp_forge.validation.stubs import (
    FilesystemStub,
    MCPStub,
    StubRegistry,
    WeatherStub,
)


class TestWeatherStub:
    """Tests for WeatherStub class."""

    def test_is_mcp_stub(self):
        """WeatherStub inherits from MCPStub."""
        stub = WeatherStub()
        assert isinstance(stub, MCPStub)

    def test_deterministic_same_seed(self):
        """Same seed produces identical results."""
        stub1 = WeatherStub(seed=42)
        stub2 = WeatherStub(seed=42)

        result1 = stub1.execute("get_weather", {"location": "Paris"})
        result2 = stub2.execute("get_weather", {"location": "Paris"})

        assert result1 == result2

    def test_different_seed_same_result(self):
        """Weather data is static, so seed doesn't affect results."""
        stub1 = WeatherStub(seed=42)
        stub2 = WeatherStub(seed=123)

        result1 = stub1.execute("get_weather", {"location": "Paris"})
        result2 = stub2.execute("get_weather", {"location": "Paris"})

        # Weather data is deterministic regardless of seed
        assert result1 == result2

    def test_returns_valid_response(self):
        """Weather stub returns expected format."""
        stub = WeatherStub()
        result = stub.execute("get_weather", {"location": "Paris"})

        assert "location" in result
        assert "temperature" in result
        assert "units" in result
        assert "condition" in result
        assert result["location"] == "Paris"
        assert result["temperature"] == 18
        assert result["condition"] == "cloudy"

    def test_celsius_default(self):
        """Default units is celsius."""
        stub = WeatherStub()
        result = stub.execute("get_weather", {"location": "Tokyo"})
        assert result["units"] == "celsius"
        assert result["temperature"] == 22

    def test_fahrenheit_conversion(self):
        """Fahrenheit units converts temperature."""
        stub = WeatherStub()
        result = stub.execute("get_weather", {"location": "Paris", "units": "fahrenheit"})
        assert result["units"] == "fahrenheit"
        # 18 * 9/5 + 32 = 64.4 -> 64
        assert result["temperature"] == 64

    def test_unknown_location(self):
        """Unknown location returns default weather."""
        stub = WeatherStub()
        result = stub.execute("get_weather", {"location": "UnknownCity"})
        assert result["location"] == "UnknownCity"
        assert result["temperature"] == 20
        assert result["condition"] == "unknown"

    def test_unknown_tool_returns_error(self):
        """Unknown tool name returns error."""
        stub = WeatherStub()
        result = stub.execute("unknown_tool", {})
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_get_tools_returns_schema(self):
        """get_tools returns valid tool definition."""
        stub = WeatherStub()
        tools = stub.get_tools()

        assert len(tools) == 1
        tool = tools[0]
        assert tool["name"] == "get_weather"
        assert "description" in tool
        assert "inputSchema" in tool
        assert tool["inputSchema"]["required"] == ["location"]

    def test_call_count_increments(self):
        """Call count increments on each execute."""
        stub = WeatherStub()
        assert stub.call_count == 0

        stub.execute("get_weather", {"location": "Paris"})
        assert stub.call_count == 1

        stub.execute("get_weather", {"location": "Tokyo"})
        assert stub.call_count == 2

    def test_known_cities(self):
        """All known cities return correct weather."""
        stub = WeatherStub()
        cities = {
            "Paris": (18, "cloudy"),
            "London": (14, "rainy"),
            "Tokyo": (22, "sunny"),
            "New York": (16, "partly_cloudy"),
        }
        for city, (temp, condition) in cities.items():
            result = stub.execute("get_weather", {"location": city})
            assert result["temperature"] == temp
            assert result["condition"] == condition


class TestFilesystemStub:
    """Tests for FilesystemStub class."""

    def test_is_mcp_stub(self):
        """FilesystemStub inherits from MCPStub."""
        stub = FilesystemStub()
        assert isinstance(stub, MCPStub)

    def test_deterministic_same_seed(self):
        """Same seed produces identical results."""
        stub1 = FilesystemStub(seed=42)
        stub2 = FilesystemStub(seed=42)

        result1 = stub1.execute("list_files", {"path": "/home/user/documents"})
        result2 = stub2.execute("list_files", {"path": "/home/user/documents"})

        assert result1 == result2

    def test_returns_valid_response(self):
        """Filesystem stub returns expected format."""
        stub = FilesystemStub()
        result = stub.execute("list_files", {"path": "/home/user/documents"})

        assert "path" in result
        assert "files" in result
        assert result["path"] == "/home/user/documents"
        assert result["files"] == ["report.pdf", "notes.txt"]

    def test_projects_directory(self):
        """Projects directory returns correct files."""
        stub = FilesystemStub()
        result = stub.execute("list_files", {"path": "/home/user/projects"})
        assert result["files"] == ["app.py", "test.py", "README.md"]

    def test_unknown_path(self):
        """Unknown path returns empty list."""
        stub = FilesystemStub()
        result = stub.execute("list_files", {"path": "/nonexistent"})
        assert result["path"] == "/nonexistent"
        assert result["files"] == []

    def test_unknown_tool_returns_error(self):
        """Unknown tool name returns error."""
        stub = FilesystemStub()
        result = stub.execute("read_file", {"path": "/test"})
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_get_tools_returns_schema(self):
        """get_tools returns valid tool definition."""
        stub = FilesystemStub()
        tools = stub.get_tools()

        assert len(tools) == 1
        tool = tools[0]
        assert tool["name"] == "list_files"
        assert "description" in tool
        assert "inputSchema" in tool
        assert tool["inputSchema"]["required"] == ["path"]

    def test_call_count_increments(self):
        """Call count increments on each execute."""
        stub = FilesystemStub()
        assert stub.call_count == 0

        stub.execute("list_files", {"path": "/home/user/documents"})
        assert stub.call_count == 1


class TestStubRegistry:
    """Tests for StubRegistry class."""

    def test_get_weather_stub(self):
        """Registry returns WeatherStub for 'weather'."""
        stub = StubRegistry.get("weather")
        assert isinstance(stub, WeatherStub)

    def test_get_filesystem_stub(self):
        """Registry returns FilesystemStub for 'filesystem'."""
        stub = StubRegistry.get("filesystem")
        assert isinstance(stub, FilesystemStub)

    def test_get_with_seed(self):
        """Registry passes seed to stub."""
        stub = StubRegistry.get("weather", seed=123)
        assert stub.rng.random() == WeatherStub(seed=123).rng.random()

    def test_unknown_stub_raises(self):
        """Registry raises ValueError for unknown stub type."""
        with pytest.raises(ValueError, match="Unknown stub"):
            StubRegistry.get("unknown")

    def test_error_message_lists_available(self):
        """Error message includes list of available stubs."""
        with pytest.raises(ValueError) as exc_info:
            StubRegistry.get("database")
        error_msg = str(exc_info.value)
        assert "weather" in error_msg
        assert "filesystem" in error_msg

    def test_available_returns_stub_types(self):
        """available() returns list of stub types."""
        available = StubRegistry.available()
        assert "weather" in available
        assert "filesystem" in available
        assert len(available) == 2
