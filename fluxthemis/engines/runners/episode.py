"""Episode planning and execution."""

from __future__ import annotations

import ctypes
import os
import signal
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable

import numpy as np

from fluxthemis.environments.base import BaseEnvironment
from fluxthemis.model_clients.base import BasePolicyClient
from fluxthemis.structures import (ActionChunk, EpisodeResult, StepResult,
                                   TaskSpec)

from .video import EpisodeVideoRecorder


@dataclass(frozen=True)
class EpisodeJob:
    """One canonical episode scheduled onto a simulator worker.

    Seeds control policy and simulator randomness.  The environment case index
    selects a task-local initial condition such as a LIBERO state or RoboDojo
    layout.
    """

    global_episode_number: int
    task: TaskSpec
    total_tasks: int
    episode_index: int
    episodes_per_task: int
    policy_seed: int
    environment_seed: int | None
    environment_case_index: int | None
    episode_id: str

    def episode_start_payload(self, started_at: str) -> dict[str, Any]:
        return {
            'global_episode_number': self.global_episode_number,
            'task_id': self.task.task_id,
            'total_tasks': self.total_tasks,
            'episode_index': self.episode_index,
            'task_episode_number': self.episode_index + 1,
            'episodes_per_task': self.episodes_per_task,
            'policy_seed': self.policy_seed,
            'environment_seed': self.environment_seed,
            'environment_case_index': self.environment_case_index,
            'episode_id': self.episode_id,
            'started_at': started_at,
        }

    def validate_result(self, result: EpisodeResult) -> None:
        identity = (
            result.task_id,
            result.episode_index,
            result.policy_seed,
            result.environment_seed,
            result.environment_case_index,
        )
        expected = (
            self.task.task_id,
            self.episode_index,
            self.policy_seed,
            self.environment_seed,
            self.environment_case_index,
        )
        if identity != expected:
            raise RuntimeError(
                f'episode result identity {identity!r} does not match job '
                f'{expected!r}')


def select_tasks(task_manifest: Sequence[TaskSpec],
                 task_ids: Sequence[str] | None) -> tuple[TaskSpec, ...]:
    """Validate an environment manifest and apply an optional task filter."""
    tasks = tuple(TaskSpec.from_value(task) for task in task_manifest)
    if not tasks:
        raise ValueError('Environment exposes no evaluation tasks')
    available_ids = [task.task_id for task in tasks]
    if len(available_ids) != len(set(available_ids)):
        raise ValueError(
            f'Environment returned duplicate task ids: {available_ids}')
    if task_ids is None:
        return tasks
    tasks_by_id = {task.task_id: task for task in tasks}
    missing = [task_id for task_id in task_ids if task_id not in tasks_by_id]
    if missing:
        raise ValueError(
            f'Unknown task ids {missing}; available tasks: {available_ids}')
    return tuple(tasks_by_id[task_id] for task_id in task_ids)


def build_episode_jobs(
    tasks: Sequence[TaskSpec],
    episodes_per_task: int,
    base_seed: int,
    run_session_id: str,
    episodes_per_task_overrides: Mapping[str, int] | None = None
) -> tuple[EpisodeJob, ...]:
    """Build stable task-major jobs independent of worker completion order.

    Seeds advance globally, while environment cases restart at zero for every
    task.
    """
    overrides = episodes_per_task_overrides or {}
    jobs = []
    global_index = 0
    for task in tasks:
        task_episode_count = overrides.get(task.task_id, episodes_per_task)
        for episode_index in range(task_episode_count):
            global_episode_number = global_index + 1
            policy_seed = int(base_seed) + global_index
            episode_id = f'{run_session_id}:{global_episode_number}'
            jobs.append(
                EpisodeJob(
                    global_episode_number=global_episode_number,
                    task=task,
                    total_tasks=len(tasks),
                    episode_index=episode_index,
                    episodes_per_task=task_episode_count,
                    policy_seed=policy_seed,
                    environment_seed=policy_seed,
                    environment_case_index=episode_index,
                    episode_id=episode_id,
                ))
            global_index += 1
    return tuple(jobs)


def configure_simulator_gpu(environment: BaseEnvironment,
                            simulator_gpu_id: str | None) -> None:
    """Pin one spawned simulator process to one physical render GPU."""
    if simulator_gpu_id is None:
        return
    if not isinstance(simulator_gpu_id, str) or not simulator_gpu_id.strip():
        raise ValueError('simulator_gpu_id must be a non-empty string')
    gpu_id = simulator_gpu_id.strip()
    environment.configure_simulator_device(gpu_id)
    if not environment.requires_global_gpu_index:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    os.environ['MUJOCO_EGL_DEVICE_ID'] = gpu_id


def configure_worker_lifetime(parent_pid: int | None = None) -> None:
    """Isolate a Linux worker and terminate it if its parent disappears."""
    if sys.platform != 'linux':
        return
    if parent_pid is None:
        parent_pid = os.getppid()
    if isinstance(parent_pid, bool) or not isinstance(parent_pid, Integral):
        raise TypeError('parent_pid must be an integer')
    if parent_pid < 1:
        raise ValueError('parent_pid must be positive')

    if os.getsid(0) != os.getpid():
        os.setsid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    if os.getppid() != int(parent_pid):
        os.kill(os.getpid(), signal.SIGKILL)


def execute_episode(environment: BaseEnvironment,
                    policy_client: BasePolicyClient,
                    job: EpisodeJob,
                    max_episode_steps: int,
                    execute_horizon: int | None,
                    stop_on_success: bool,
                    on_episode_started: Callable[[], None] | None = None,
                    save_videos: bool = False,
                    video_dir: str | os.PathLike | None = None,
                    video_fps: int = 20,
                    video_worker_id: str | None = None) -> EpisodeResult:
    """Execute one episode using components local to the current process."""
    observation, reset_info = environment.reset(
        job.task,
        job.episode_index,
        environment_seed=job.environment_seed,
        environment_case_index=job.environment_case_index,
    )
    if not isinstance(reset_info, Mapping):
        raise TypeError('Environment reset info must be a mapping')
    policy_client.reset(job.task, job.episode_index, job.policy_seed)
    policy_client.set_episode_id(job.episode_id)
    if on_episode_started is not None:
        on_episode_started()

    started = time.perf_counter()
    episode_return = 0.0
    steps = 0
    model_calls = 0
    prediction_metadata: list[Mapping[str, Any]] = []
    success = False
    score: float | None = None
    termination_reason = 'max_steps'
    latest_info = dict(reset_info)
    episode_done = False
    recorder = None
    if save_videos and not environment.supports_native_video_recording:
        if video_dir is None:
            raise ValueError('video_dir is required when saving videos')
        recorder = EpisodeVideoRecorder(
            video_dir, fps=video_fps, worker_id=video_worker_id)
        recorder.start(job.task.task_id, job.episode_index)

    try:
        if recorder is not None:
            recorder.capture(environment)
        while steps < max_episode_steps and not episode_done:
            context = build_context(
                task=job.task,
                episode_index=job.episode_index,
                policy_seed=job.policy_seed,
                info=latest_info,
                episode_id=job.episode_id,
                observation=observation,
            )
            action_chunk = policy_client.predict_action(observation, context)
            actions = validate_action_chunk(action_chunk)
            if not isinstance(action_chunk.metadata, Mapping):
                raise TypeError('ActionChunk.metadata must be a mapping')
            prediction_metadata.append(dict(action_chunk.metadata))
            model_calls += 1
            horizon = (
                len(actions) if execute_horizon is None else min(
                    len(actions), execute_horizon))

            for action in actions[:horizon]:
                transition = environment.step(action.copy())
                if not isinstance(transition, StepResult):
                    raise TypeError(
                        'Environment.step() must return a StepResult')
                if not isinstance(transition.info, Mapping):
                    raise TypeError('StepResult.info must be a mapping')
                observation = transition.observation
                episode_return += float(transition.reward)
                steps += 1
                latest_info = dict(transition.info)
                if recorder is not None:
                    recorder.capture(environment)
                if transition.success is not None:
                    success = success or bool(transition.success)
                if transition.score is not None:
                    step_score = float(transition.score)
                    if not np.isfinite(step_score):
                        raise ValueError(
                            'Environment returned a non-finite episode score')
                    if not 0.0 <= step_score <= 1.0:
                        raise ValueError(
                            'Environment episode score must be in [0, 1]')
                    score = step_score

                if success and stop_on_success:
                    termination_reason = 'success'
                    episode_done = True
                elif transition.terminated:
                    termination_reason = 'terminated'
                    episode_done = True
                elif transition.truncated:
                    termination_reason = 'truncated'
                    episode_done = True
                elif steps >= max_episode_steps:
                    termination_reason = 'max_steps'
                    episode_done = True
                if episode_done:
                    break
        if recorder is not None:
            recorded_paths = recorder.finish(success)
            if recorded_paths:
                existing_paths = latest_info.get('video_paths', [])
                if isinstance(existing_paths, str):
                    existing_paths = [existing_paths]
                elif not isinstance(existing_paths, (list, tuple)):
                    raise TypeError('video_paths must be a sequence of paths')
                latest_info['video_paths'] = [*existing_paths, *recorded_paths]
    except BaseException:
        if recorder is not None:
            recorder.abort()
        raise

    return EpisodeResult(
        task_id=job.task.task_id,
        episode_index=job.episode_index,
        policy_seed=job.policy_seed,
        environment_seed=job.environment_seed,
        environment_case_index=job.environment_case_index,
        success=success,
        episode_return=episode_return,
        steps=steps,
        duration_s=time.perf_counter() - started,
        termination_reason=termination_reason,
        model_calls=model_calls,
        score=score,
        info=latest_info,
        prediction_metadata=prediction_metadata,
    )


def build_context(task: TaskSpec,
                  episode_index: int,
                  policy_seed: int,
                  info: Mapping[str, Any],
                  episode_id: str | None = None,
                  observation: Any = None) -> dict[str, Any]:
    prompt = info.get('task_description', task.description)
    if not isinstance(prompt, str):
        raise TypeError('environment task_description must be a string')
    if isinstance(observation, Mapping):
        observed_prompt = observation.get('task_description')
        if isinstance(observed_prompt, str) and observed_prompt:
            prompt = observed_prompt
    context = {
        'task_id': task.task_id,
        'task_description': prompt,
        'task_metadata': dict(task.metadata),
        'episode_index': episode_index,
        'policy_seed': policy_seed,
        'info': dict(info),
    }
    if episode_id is not None:
        context['episode_id'] = episode_id
    for key, value in task.metadata.items():
        context.setdefault(str(key), value)
    return context


def validate_action_chunk(action_chunk: ActionChunk) -> np.ndarray:
    if not isinstance(action_chunk, ActionChunk):
        raise TypeError(
            'PolicyClient.predict_action() must return an ActionChunk')
    actions = np.asarray(action_chunk.actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(
            f'PolicyClient actions must have shape [T, A], got {actions.shape}'
        )
    if actions.shape[0] == 0 or actions.shape[1] == 0:
        raise ValueError('PolicyClient returned an empty action chunk')
    if not np.isfinite(actions).all():
        raise ValueError('PolicyClient returned NaN or infinite actions')
    return actions
