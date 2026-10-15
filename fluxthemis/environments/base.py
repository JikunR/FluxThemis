from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from fluxthemis.structures import StepResult, TaskSpec


class EnvironmentCaseRejected(RuntimeError):
    """Signal that an environment case cannot start an episode."""


class BaseEnvironment(ABC):
    """Contract implemented by simulation and real-robot adapters."""

    requires_global_gpu_index = False
    supports_native_video_recording = False

    def configure_worker(self, worker_id: str, run_id: str) -> None:
        del worker_id, run_id

    def configure_simulator_device(self, device_id: str) -> None:
        del device_id

    def configure_video_recording(self, enabled: bool,
                                  output_dir: str | None) -> None:
        del enabled, output_dir

    def setup(self) -> None:
        """Allocate simulator or hardware resources."""

    @abstractmethod
    def tasks(self) -> Sequence[TaskSpec]:
        """Return the tasks available for evaluation."""

    @abstractmethod
    def reset(self, task: TaskSpec, episode_index: int, *,
              environment_seed: int | None,
              environment_case_index: int | None) -> tuple[Any, dict]:
        """Reset one episode and return ``(observation, info)``."""

    @abstractmethod
    def step(self, action: Any) -> StepResult:
        """Execute one environment-unit action."""

    def render(self) -> Any | None:
        return None

    def close(self) -> None:
        """Release resources. Implementations should make this idempotent."""
