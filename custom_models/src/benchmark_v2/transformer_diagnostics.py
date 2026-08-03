from __future__ import annotations

from typing import Any, Callable

from .nonfinite import nested_tensor_summaries


class TransformerHookCapture:
    """Bounded forward-hook capture used only for non-finite diagnostics."""

    def __init__(self, model: Any, recorder: Callable[[dict[str, Any]], None]):
        self.model = model
        self.recorder = recorder
        self._handles: list[Any] = []

    def __enter__(self) -> "TransformerHookCapture":
        for name, module in self.model.named_modules():
            if not name:
                continue

            def hook(_module, inputs, output, module_name=name):
                input_summaries = nested_tensor_summaries(inputs, "input")
                output_summaries = nested_tensor_summaries(output, "output")
                self.recorder(
                    {
                        "module_name": module_name,
                        "module_class": type(_module).__name__,
                        "inputs": list(input_summaries.values()),
                        "outputs": list(output_summaries.values()),
                    }
                )

            self._handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
