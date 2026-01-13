"""ValidationRunner class for executing validation loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from mcp_forge.data.formatter import parse_tool_call
from mcp_forge.state import ToolDefinition, ValidationResult
from mcp_forge.tools.inspector import validate_tool_call
from mcp_forge.validation.config import ValidationConfig
from mcp_forge.validation.stubs import MCPStub, StubRegistry


@dataclass
class ValidationSample:
    """A single validation test case."""

    prompt: str
    expected_tool: str | None
    expected_args: dict[str, Any] | None


class ValidationRunner:
    """Executes validation loops against trained models."""

    def __init__(self, config: ValidationConfig, tools: list[ToolDefinition]) -> None:
        """Initialize validation runner.

        Args:
            config: Validation configuration
            tools: List of tool definitions for validation
        """
        self.config = config
        self.tools = tools
        self.model = None
        self.tokenizer = None
        self._stub: MCPStub | None = None

    def load_model(self) -> None:
        """Load the trained model (LoRA adapter).

        Raises:
            NotImplementedError: If model_path points to a GGUF file
            RuntimeError: If model loading fails
        """
        # Import here to avoid heavy dependencies at module level
        from unsloth import FastLanguageModel

        model_path = self.config.model_path

        # Detect if GGUF or LoRA adapter
        if model_path.suffix == ".gguf":
            raise NotImplementedError("GGUF validation not yet implemented")

        # Load LoRA adapter
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        # Enable inference mode
        FastLanguageModel.for_inference(self.model)

    def _get_stub(self) -> MCPStub:
        """Get or create MCP stub.

        Returns:
            MCPStub instance

        Raises:
            ValueError: If no stub configuration provided
        """
        if self._stub is None:
            if self.config.stub_config is None:
                raise ValueError("No stub configuration provided")
            self._stub = StubRegistry.get(
                self.config.stub_config.stub_type,
                seed=self.config.stub_config.seed,
            )
        return self._stub

    def generate_response(self, prompt: str) -> str:
        """Generate model response for a prompt.

        Args:
            prompt: User prompt to respond to

        Returns:
            Model-generated response string

        Raises:
            RuntimeError: If model not loaded
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model must be loaded before generating responses")

        # Build system prompt with tools
        from mcp_forge.data.formatter import format_system_prompt

        system_prompt = format_system_prompt(self.tools)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            inputs,
            max_new_tokens=self.config.inference.max_new_tokens,
            temperature=self.config.inference.temperature,
            top_p=self.config.inference.top_p,
            do_sample=self.config.inference.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode only the generated part
        response = self.tokenizer.decode(
            outputs[0][inputs.shape[1] :],
            skip_special_tokens=True,
        )

        return response

    def validate_single(
        self,
        sample: ValidationSample,
    ) -> dict[str, Any]:
        """Validate a single sample and return metrics.

        Args:
            sample: Validation sample to test

        Returns:
            Dictionary with validation results for this sample
        """
        result: dict[str, Any] = {
            "prompt": sample.prompt,
            "parsed": False,
            "schema_valid": False,
            "tool_correct": False,
            "loop_complete": False,
            "error": None,
        }

        try:
            # Generate response
            response = self.generate_response(sample.prompt)
            result["response"] = response

            # Parse tool call
            tool_call = parse_tool_call(response)
            if tool_call is None:
                if sample.expected_tool is None:
                    # Correctly identified no tool needed
                    result["parsed"] = True
                    result["schema_valid"] = True
                    result["tool_correct"] = True
                    result["loop_complete"] = True
                else:
                    result["error"] = "No tool call found in response"
                return result

            result["parsed"] = True
            result["tool_call"] = tool_call

            # Validate schema
            is_valid, error = validate_tool_call(tool_call, self.tools)
            result["schema_valid"] = is_valid
            if not is_valid:
                result["error"] = error
                return result

            # Check tool selection
            if sample.expected_tool:
                result["tool_correct"] = tool_call["name"] == sample.expected_tool
            else:
                result["tool_correct"] = True  # Any valid tool is acceptable

            # Execute against stub/server
            if self.config.stub_config:
                stub = self._get_stub()
                exec_result = stub.execute(
                    tool_call["name"], tool_call.get("arguments", {})
                )
                result["execution_result"] = exec_result
                result["loop_complete"] = "error" not in exec_result
            else:
                # Real MCP server execution
                result["loop_complete"] = self._execute_on_real_server(tool_call)

        except Exception as e:
            result["error"] = str(e)

        return result

    def _execute_on_real_server(self, tool_call: dict[str, Any]) -> bool:
        """Execute tool call on real MCP server with retries.

        Args:
            tool_call: Tool call dictionary with name and arguments

        Returns:
            True if execution succeeded, False otherwise
        """
        import asyncio
        import time

        from mcp_forge.tools.inspector import inspect_mcp_server

        # TODO: Implement real MCP tool execution
        # For now, just verify server is reachable
        for attempt in range(self.config.retry_count):
            try:
                asyncio.run(
                    inspect_mcp_server(
                        self.config.mcp_command,
                        timeout=self.config.timeout,
                    )
                )
                return True
            except Exception:
                if attempt < self.config.retry_count - 1:
                    time.sleep(self.config.retry_delay)
        return False

    def run(
        self,
        samples: list[ValidationSample],
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> ValidationResult:
        """Run full validation and return aggregated results.

        Args:
            samples: List of validation samples to test
            progress_callback: Optional callback(current, total, progress_pct)

        Returns:
            ValidationResult with aggregated metrics
        """
        if self.model is None:
            self.load_model()

        total = len(samples)
        results: list[dict[str, Any]] = []

        for i, sample in enumerate(samples):
            result = self.validate_single(sample)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total, (i + 1) / total)

        # Aggregate metrics
        parsed = sum(1 for r in results if r["parsed"])
        schema_valid = sum(1 for r in results if r["schema_valid"])
        tool_correct = sum(1 for r in results if r["tool_correct"])
        loop_complete = sum(1 for r in results if r["loop_complete"])

        failures = [r for r in results if r.get("error")]

        validation_result = ValidationResult(
            passed=all(
                [
                    parsed / total >= self.config.parse_threshold if total > 0 else True,
                    schema_valid / total >= self.config.schema_threshold
                    if total > 0
                    else True,
                    tool_correct / total >= self.config.accuracy_threshold
                    if total > 0
                    else True,
                    loop_complete / total >= self.config.loop_threshold
                    if total > 0
                    else True,
                ]
            ),
            samples_tested=total,
            samples_passed=sum(
                1 for r in results if r["loop_complete"] and not r.get("error")
            ),
            tool_call_parse_rate=parsed / total if total > 0 else 0.0,
            schema_conformance_rate=schema_valid / total if total > 0 else 0.0,
            tool_selection_accuracy=tool_correct / total if total > 0 else 0.0,
            loop_completion_rate=loop_complete / total if total > 0 else 0.0,
            failures=[{"sample": f["prompt"], "error": f["error"]} for f in failures],
        )

        return validation_result


def generate_validation_samples(
    tools: list[ToolDefinition],
    count: int = 20,
    include_no_tool: bool = True,
    seed: int = 42,
) -> list[ValidationSample]:
    """Generate validation samples from tool definitions.

    Creates diverse test prompts for each tool, ensuring coverage
    of different argument combinations and edge cases.

    Args:
        tools: Tool definitions to generate samples for
        count: Total number of samples to generate
        include_no_tool: Whether to include no-tool test cases
        seed: Random seed for reproducibility

    Returns:
        List of ValidationSample objects
    """
    import random

    rng = random.Random(seed)

    samples: list[ValidationSample] = []

    # Templates for generating prompts
    WEATHER_PROMPTS = [
        "What's the weather in {location}?",
        "Is it going to rain in {location} today?",
        "Tell me the current temperature in {location}.",
        "What should I wear in {location} based on the weather?",
    ]

    FILESYSTEM_PROMPTS = [
        "List the files in {path}.",
        "What files are in the {path} directory?",
        "Show me what's inside {path}.",
    ]

    NO_TOOL_PROMPTS = [
        "What is the capital of France?",
        "How do I make pasta?",
        "Tell me a joke.",
        "What year did World War II end?",
    ]

    # Generate tool-specific samples
    samples_per_tool = count // len(tools) if tools else 0

    for tool in tools:
        if tool.name == "get_weather":
            locations = ["Paris", "London", "Tokyo", "New York", "Sydney"]
            for i in range(samples_per_tool):
                location = rng.choice(locations)
                prompt = rng.choice(WEATHER_PROMPTS).format(location=location)
                samples.append(
                    ValidationSample(
                        prompt=prompt,
                        expected_tool="get_weather",
                        expected_args={"location": location},
                    )
                )

        elif tool.name == "list_files":
            paths = ["/home/user/documents", "/home/user/projects", "/tmp"]
            for i in range(samples_per_tool):
                path = rng.choice(paths)
                prompt = rng.choice(FILESYSTEM_PROMPTS).format(path=path)
                samples.append(
                    ValidationSample(
                        prompt=prompt,
                        expected_tool="list_files",
                        expected_args={"path": path},
                    )
                )

        else:
            # Generic sample generation for unknown tools
            for i in range(samples_per_tool):
                samples.append(
                    ValidationSample(
                        prompt=f"Use the {tool.name} tool with default arguments.",
                        expected_tool=tool.name,
                        expected_args={},
                    )
                )

    # Add no-tool samples
    if include_no_tool:
        no_tool_count = max(1, count // 5)  # ~20% no-tool
        for prompt in rng.sample(
            NO_TOOL_PROMPTS, min(no_tool_count, len(NO_TOOL_PROMPTS))
        ):
            samples.append(
                ValidationSample(
                    prompt=prompt,
                    expected_tool=None,
                    expected_args=None,
                )
            )

    rng.shuffle(samples)
    return samples[:count]
