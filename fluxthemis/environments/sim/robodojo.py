from __future__ import annotations

import argparse
import contextlib
import io
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fluxthemis.engines.utils import ENVIRONMENTS
from fluxthemis.structures import StepResult, TaskSpec

from ..base import BaseEnvironment, EnvironmentCaseRejected

ROBODOJO_BENCHMARK = 'RoboDojo'
ROBODOJO_DEFAULT_ROOT = '/root/projects/RoboDojo'
ROBODOJO_ACTION_MODES = ('joint', 'ee')
ROBODOJO_EE_POSE_DIM = 7
ROBODOJO_REQUIRED_SUBDIRS = ('env', 'env_cfg', 'src', 'task', 'XPolicyLab')
ROBODOJO_ALL_TASKS = 'all'
ROBODOJO_RANDOM_SUFFIX = '_random'

# Capability groups used by the RoboDojo leaderboard plus the runnable
# ``*_random`` generalization variants. Keep the base task names in sync with
# RoboDojo's ``scripts/internal/task_inventory.py`` and
# ``scripts/internal/summarize_result.py``.
ROBODOJO_DIMENSION_TASKS = {
    'generalization':
    ('stack_bowls', 'push_T', 'pack_objects_into_box', 'fold_clothes',
     'hang_mugs', 'sweep_blocks', 'pour_liquid_into_cup', 'make_toast',
     'arrange_largest_number', 'sort_nesting_dolls_by_size',
     'store_laptop_and_headphones', 'stack_blocks', 'stack_bowls_random',
     'push_T_random', 'pack_objects_into_box_random', 'fold_clothes_random',
     'hang_mugs_random', 'sweep_blocks_random', 'pour_liquid_into_cup_random',
     'make_toast_random', 'arrange_largest_number_random',
     'sort_nesting_dolls_by_size_random', 'store_laptop_and_headphones_random',
     'stack_blocks_random'),
    'memory': ('cover_blocks', 'match_and_pick_from_conveyor', 'swap_blocks',
               'swap_T', 'press_by_number', 'imitate_sorting_sequence'),
    'precision': ('fasten_screws', 'plug_in_charger', 'insert_tubes',
                  'pour_balls_into_vase', 'play_Xylophone', 'deposit_coin',
                  'insert_key', 'build_tower'),
    'long-horizon':
    ('fill_pen_holder', 'classify_objects', 'put_bottles_into_dustbin',
     'play_tic_tac_toe', 'fill_egg_holder', 'organize_table', 'make_kong',
     'play_stacking_toy'),
    'open': ('align_blocks', 'general_pickup', 'solve_equation',
             'stack_blocks_by_language', 'classify_objects_by_language',
             'pick_from_conveyor_by_image', 'store_tools_in_toolbox',
             'pour_by_language'),
}

# Flat transport layout matching themis.transport.image_keys / state_keys:
# three top-level camera images plus the 14D joint state as ``states``.
FLAT_CAMERA_KEYS = (('cam_high', 'cam_head'),
                    ('cam_left_wrist', 'cam_left_wrist'), ('cam_right_wrist',
                                                           'cam_right_wrist'))
FLAT_STATE_KEYS = ('left_arm_joint_state', 'left_ee_joint_state',
                   'right_arm_joint_state', 'right_ee_joint_state')
FLAT_EE_POSE_KEYS = ('left_ee_pose', 'right_ee_pose')


def robodojo_obs_to_flat(obs: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a nested RoboDojo observation to the flat transport layout.

    Produces top-level ``cam_high`` / ``cam_left_wrist`` /
    ``cam_right_wrist`` images, a 14D ``states`` vector (left arm first),
    per-arm world-frame end-effector poses and gripper positions, and
    ``task_description``, so the observation matches ``image_keys`` /
    ``state_keys`` declared in ``themis.transport``.  The explicit pose fields
    allow checkpoint-free Cartesian controllers to use RoboDojo's ``ee``
    action mode without reverse kinematics from joint observations.
    """
    vision = obs.get('vision')
    if not isinstance(vision, Mapping):
        raise ValueError("RoboDojo observation missing 'vision' mapping")
    flat: dict[str, Any] = {}
    for flat_key, camera_key in FLAT_CAMERA_KEYS:
        camera = vision.get(camera_key)
        if not isinstance(camera, Mapping) or 'color' not in camera:
            raise ValueError(
                f"RoboDojo observation missing vision['{camera_key}']"
                "['color']")
        flat[flat_key] = np.asarray(camera['color'])
    state = obs.get('state')
    if not isinstance(state, Mapping):
        raise ValueError("RoboDojo observation missing 'state' mapping")
    parts = []
    for key in FLAT_STATE_KEYS:
        if key not in state:
            raise KeyError(f"Missing key '{key}' in obs['state']")
        parts.append(np.asarray(state[key], dtype=np.float32))
    flat['states'] = np.concatenate(parts)
    for key in FLAT_EE_POSE_KEYS:
        if key not in state:
            raise KeyError(f"Missing key '{key}' in obs['state']")
        flat[key] = np.asarray(state[key], dtype=np.float32).copy()
    flat['left_gripper_qpos'] = np.asarray(
        state['left_ee_joint_state'], dtype=np.float32).copy()
    flat['right_gripper_qpos'] = np.asarray(
        state['right_ee_joint_state'], dtype=np.float32).copy()
    instruction = obs.get('instruction')
    if isinstance(instruction, str) and instruction:
        flat['task_description'] = instruction
    return flat


def robodojo_task_inventory(
        robodojo_root: str | os.PathLike) -> tuple[str, ...]:
    """Return runnable RoboDojo task names without importing the stack."""
    root = Path(robodojo_root)
    config_dir = root / 'task' / ROBODOJO_BENCHMARK / 'config'
    tasks_dir = root / 'task' / ROBODOJO_BENCHMARK / 'tasks'
    tasks = []
    if config_dir.is_dir() and tasks_dir.is_dir():
        for path in sorted(config_dir.glob('*.yml')):
            if path.stem == '_task':
                continue
            if (tasks_dir / f'{path.stem}.py').is_file():
                tasks.append(path.stem)
    return tuple(tasks)


def _coerce_task_names(task_name: str | Sequence[str]) -> list[str]:
    """Normalize ``task_name`` into an ordered list of non-empty names."""
    if isinstance(task_name, str):
        values = [
            value.strip() for value in task_name.split(',') if value.strip()
        ]
    else:
        try:
            values = [str(value).strip() for value in task_name]
        except TypeError as exc:
            raise TypeError(
                'task_name must be a string or a sequence of strings') from exc
    if not values:
        raise ValueError('task_name cannot be empty')
    if any(not value for value in values):
        raise ValueError('task_name cannot contain empty names')
    return values


def _task_dimension(task_name: str) -> str | None:
    """Return the leaderboard capability dimension for a task name."""
    base_name = task_name
    if base_name.endswith(ROBODOJO_RANDOM_SUFFIX):
        base_name = base_name[:-len(ROBODOJO_RANDOM_SUFFIX)]
    for dimension, tasks in ROBODOJO_DIMENSION_TASKS.items():
        if base_name in tasks:
            return dimension
    return None


def _is_random_task(task_name: str) -> bool:
    return task_name.endswith(ROBODOJO_RANDOM_SUFFIX)


def _is_recoverable_robodojo_case_error(exc: BaseException) -> bool:
    """Return whether an exception means ``retry with another layout``.

    RoboDojo raises ``UnStableError`` for physically unstable pre-generated
    layouts and ``PhysXBrokenError`` when an active simulation develops an
    invalid transform. Unexpected failures must still surface normally.
    """
    return type(exc).__name__ in {'PhysXBrokenError', 'UnStableError'}


def _task_metadata(task_name: str, task_index: int) -> dict[str, Any]:
    dimension = _task_dimension(task_name)
    is_random = _is_random_task(task_name)
    base_task = task_name
    if is_random:
        base_task = task_name[:-len(ROBODOJO_RANDOM_SUFFIX)]
    metadata: dict[str, Any] = {
        'benchmark': 'robodojo',
        'task_name': task_name,
        'task_index': task_index,
    }
    if dimension is not None:
        metadata['dimension'] = dimension
        metadata['variant'] = 'random' if is_random else 'base'
        metadata['base_task'] = base_task
    return metadata


def _resolve_task_names(task_name: str | Sequence[str],
                        inventory: Sequence[str]) -> tuple[str, ...]:
    """Expand aliases/dimensions and validate task names against inventory."""
    values = _coerce_task_names(task_name)
    resolved: list[str] = []
    for value in values:
        dimension = value.lower().replace('_', '-')
        if value == ROBODOJO_ALL_TASKS:
            all_tasks = tuple(task
                              for tasks in ROBODOJO_DIMENSION_TASKS.values()
                              for task in tasks)
            if inventory:
                all_tasks = tuple(task for task in all_tasks
                                  if task in inventory)
            if not all_tasks:
                raise ValueError(
                    'task_name=all requires a non-empty RoboDojo task '
                    'inventory')
            resolved.extend(all_tasks)
        elif dimension in ROBODOJO_DIMENSION_TASKS:
            dimension_tasks = ROBODOJO_DIMENSION_TASKS[dimension]
            if inventory:
                dimension_tasks = tuple(task for task in dimension_tasks
                                        if task in inventory)
            if not dimension_tasks:
                raise ValueError(
                    f'RoboDojo dimension {dimension!r} contains no runnable '
                    f'tasks from {sorted(inventory)!r}')
            resolved.extend(dimension_tasks)
        else:
            resolved.append(value)

    seen: set[str] = set()
    unique = []
    for value in resolved:
        if value not in seen:
            seen.add(value)
            unique.append(value)

    if inventory:
        missing = sorted({value for value in unique if value not in inventory})
        if missing:
            raise ValueError(
                f'Unsupported RoboDojo task(s) {missing!r}; expected one of '
                f'{sorted(inventory)}')
    return tuple(unique)


@dataclass(frozen=True)
class _PrelaunchRuntime:
    AppLauncher: Any
    OmegaConf: Any
    get_physx_monitor: Any
    ROOT_DIR: str
    ENV_CONFIG_PATH: str
    BENCHMARK: str
    load_yaml: Any
    process_config: Any
    process_randomization: Any


@dataclass(frozen=True)
class _PostlaunchRuntime:
    create_eval_env: Any
    task_registry: Any


def _load_prelaunch_runtime() -> _PrelaunchRuntime:
    """Import the modules that are safe before the Isaac Sim app launches."""
    try:
        from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
        from isaaclab.app import AppLauncher
        from omegaconf import OmegaConf
        from src.eval_client.physx_warning_monitor import get_monitor
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'RoboDojoEnvironment requires the RoboDojo simulation stack. '
            'Install it with `bash scripts/install_robodojo.sh`.') from exc
    return _PrelaunchRuntime(
        AppLauncher=AppLauncher,
        OmegaConf=OmegaConf,
        get_physx_monitor=get_monitor,
        ROOT_DIR=ROOT_DIR,
        ENV_CONFIG_PATH=ENV_CONFIG_PATH,
        BENCHMARK=BENCHMARK,
        load_yaml=load_yaml,
        process_config=process_config,
        process_randomization=process_randomization,
    )


def _load_postlaunch_runtime() -> _PostlaunchRuntime:
    """Import the Isaac-app-dependent modules after the app has launched."""
    try:
        from src.eval_client.eval_env import create_eval_env
        from task.RoboDojo import task_registry
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'RoboDojoEnvironment requires the RoboDojo simulation stack. '
            'Install it with `bash scripts/install_robodojo.sh`.') from exc
    return _PostlaunchRuntime(
        create_eval_env=create_eval_env,
        task_registry=task_registry,
    )


@ENVIRONMENTS.register_module()
class RoboDojoEnvironment(BaseEnvironment):
    """RoboDojo (Isaac Sim) simulation adapter.

    RoboDojo's native runner is one-process-per-task, so switching tasks in
    FluxThemis recreates the RoboDojo ``EvalEnv`` while reusing the same Isaac
    Sim app. The adapter can expose one task, a comma-separated task list, a
    capability dimension such as ``generalization``, or ``all``.

    Like the LIBERO and RoboCasa adapters, it returns observations in the flat
    transport layout declared by ``themis.transport.image_keys`` /
    ``state_keys``: top-level ``cam_high`` / ``cam_left_wrist`` /
    ``cam_right_wrist`` images plus a 14D ``states`` vector and
    ``task_description`` (see :func:`robodojo_obs_to_flat`). It converts a flat
    environment-unit action row ``[A]`` into the RoboDojo action mapping.
    Image selection, resizing, state conversion and action denormalization
    remain owned by the FluxVLA server-side dataset and transforms.
    """

    requires_global_gpu_index = True
    supports_native_video_recording = True

    def __init__(self,
                 task_name: str | Sequence[str],
                 env_cfg_type: str = 'arx_x5',
                 robodojo_root: str | os.PathLike | None = None,
                 action_mode: str = 'joint',
                 device_id: int | None = None,
                 headless: bool = True,
                 num_envs: int = 1,
                 seed: int = 0,
                 save_videos: bool = False,
                 video_dir: str | os.PathLike | None = None) -> None:
        if not isinstance(env_cfg_type, str) or not env_cfg_type:
            raise ValueError('env_cfg_type must be a non-empty string')
        if action_mode not in ROBODOJO_ACTION_MODES:
            raise ValueError(
                f'Unsupported RoboDojo action_mode {action_mode!r}; expected '
                f'one of {ROBODOJO_ACTION_MODES}')
        if device_id is not None:
            device_id = self._nonnegative_integer(device_id, 'device_id')
        if not isinstance(headless, bool):
            raise TypeError('headless must be a boolean')
        if not isinstance(save_videos, bool):
            raise TypeError('save_videos must be a boolean')
        num_envs = self._nonnegative_integer(num_envs, 'num_envs')
        if num_envs != 1:
            raise ValueError(
                'RoboDojoEnvironment currently drives a single environment; '
                'num_envs must be 1')
        seed = self._nonnegative_integer(seed, 'seed')

        root_value = robodojo_root or os.environ.get(
            'ROBODOJO_ROOT') or ROBODOJO_DEFAULT_ROOT
        self.robodojo_root = str(Path(root_value).expanduser().resolve())
        self.env_cfg_type = env_cfg_type
        self.action_mode = action_mode
        self.device_id = device_id
        self.headless = headless
        self.num_envs = int(num_envs)
        self.seed = int(seed)
        self.worker_id: str | None = None
        self.run_id: str | None = None
        self.save_videos = save_videos
        if video_dir is None:
            self._video_dir: Path | None = None
        else:
            if not isinstance(video_dir, (str, os.PathLike)):
                raise TypeError('video_dir must be a path or None')
            video_path = Path(video_dir).expanduser()
            if not video_path.is_absolute():
                video_path = video_path.resolve()
            if not str(video_path):
                raise ValueError('video_dir cannot be empty')
            self._video_dir = video_path
        inventory = robodojo_task_inventory(self.robodojo_root)
        task_names = _resolve_task_names(task_name, inventory)
        self._tasks = tuple(
            TaskSpec(
                task_id=task_id,
                metadata=_task_metadata(task_id, task_index),
            ) for task_index, task_id in enumerate(task_names))
        self._task_ids = {task.task_id for task in self._tasks}
        self._task = self._tasks[0]

        env_cfg_path = Path(
            self.robodojo_root) / 'env_cfg' / (f'{env_cfg_type}.yml')
        if env_cfg_path.is_file():
            config = _read_env_cfg_config(env_cfg_path)
            if 'config' not in config:
                raise ValueError(
                    f'RoboDojo env config {env_cfg_type!r} must define a '
                    '`config` section')

        self._env = None
        self._simulation_app = None
        self._runtime: _PrelaunchRuntime | None = None
        self._post: _PostlaunchRuntime | None = None
        self._physx_monitor = None
        self._original_cwd: str | None = None
        self._episode_active = False
        self._episode_task: TaskSpec | None = None
        self._episode_index: int | None = None
        self._episode_layout_index: int | None = None
        self._video_finalized = True

    def setup(self) -> None:
        if self._env is not None:
            raise RuntimeError('RoboDojoEnvironment.setup() was called twice')
        self._prepare_runtime_paths()
        self._configure_run_identity()
        runtime = _load_prelaunch_runtime()
        physx_monitor = runtime.get_physx_monitor()
        physx_monitor.start(enabled=True)
        self._physx_monitor = physx_monitor
        app_launcher = runtime.AppLauncher(
            self._build_app_args(runtime.AppLauncher))
        simulation_app = app_launcher.app
        self._runtime = runtime
        self._simulation_app = simulation_app
        try:
            # Isaac Sim / Kit may prepend its own entries to sys.path after
            # launch. Restore the RoboDojo root to sys.path[0] before any
            # post-launch RoboDojo module import, otherwise XPolicyLab's
            # top-level ``utils`` package can shadow the simulator's.
            self._sync_runtime_paths()
            post = _load_postlaunch_runtime()
            # ``create_eval_env`` constructs RoboDojo's ``WsModelClient``
            # synchronously, which immediately starts a 15-minute retry loop
            # against ``ws://localhost:0``. FluxThemis drives the environment
            # directly over its own ZMQ model client, so replace the
            # WebSocket client with the same no-op stand-in assigned below.
            post.create_eval_env.__globals__[
                'WsModelClient'] = _NoOpModelClient
            self._post = post
            self._activate_task(self._tasks[0])
        except Exception:
            try:
                simulation_app.close()
            except Exception:
                pass
            self._runtime = None
            self._post = None
            self._simulation_app = None
            self._env = None
            physx_monitor.shutdown()
            self._physx_monitor = None
            raise

    def _create_env_for_active_task(self) -> None:
        if self._runtime is None or self._post is None:
            raise RuntimeError('RoboDojo runtime was not initialized')
        if self._simulation_app is None:
            raise RuntimeError('RoboDojo simulation app was not initialized')
        self._sync_runtime_paths()
        env_cfg = self._build_env_cfg(self._runtime, self._post)
        env = self._post.create_eval_env(env_cfg, self._simulation_app)
        env.model_client = _NoOpModelClient()
        if self.save_videos:
            if self._video_dir is None:
                native_save_dir = Path(env.save_dir)
                if not native_save_dir.is_absolute():
                    native_save_dir = (
                        Path(self.robodojo_root) / native_save_dir)
                self._video_dir = native_save_dir.resolve()
            self._video_dir.mkdir(parents=True, exist_ok=True)
        else:
            # ``EvalEnv.get_obs()`` otherwise streams every camera frame to
            # ``_stream/*.tmp.mp4`` even though FluxThemis never asks
            # RoboDojo's native runner to finalize it.
            env._stream_vision = lambda *args, **kwargs: None
        self._env = env

    def _activate_task(self, task: TaskSpec) -> None:
        self._finalize_active_episode('truncated')
        self._close_env_only()
        self._task = task
        self._create_env_for_active_task()

    def _close_env_only(self) -> None:
        try:
            self._close_native_env()
        finally:
            self._env = None

    def _close_native_env(self) -> None:
        env = self._env
        try:
            if env is not None:
                env.close()
        finally:
            self._episode_active = False

    def _finalize_active_episode(self, tag: str = 'truncated') -> None:
        if not self._episode_active or self._video_finalized:
            return
        self._finalize_video(self._episode_task, self._episode_index, tag)
        self._video_finalized = True

    def tasks(self) -> Sequence[TaskSpec]:
        return self._tasks

    def reset(self, task: TaskSpec, episode_index: int, *,
              environment_seed: int | None,
              environment_case_index: int | None) -> tuple[Any, dict]:
        self._require_setup()
        task = TaskSpec.from_value(task)
        if task.task_id not in self._task_ids:
            raise ValueError(
                f'RoboDojoEnvironment is configured for tasks '
                f'{sorted(self._task_ids)!r} but reset() received '
                f'{task.task_id!r}')
        if task.task_id != self._task.task_id:
            self._activate_task(task)
        else:
            self._finalize_active_episode('truncated')
            if self._episode_active:
                self._close_native_env()

        layout_index = self._nonnegative_integer(environment_case_index,
                                                 'environment_case_index')
        if self._physx_monitor is not None:
            self._physx_monitor.reset()
        try:
            self._env.reset(seed=[layout_index])
        except Exception as exc:
            if not _is_recoverable_robodojo_case_error(exc):
                raise
            self._close_native_env()
            raise EnvironmentCaseRejected(
                f'RoboDojo layout {layout_index} is unstable') from exc
        self._episode_active = True
        self._episode_task = task
        self._episode_index = int(episode_index)
        self._episode_layout_index = layout_index
        self._video_finalized = False
        # Native ``EvalEnv.run_eval()`` registers each episode's success
        # criteria before stepping. FluxThemis bypasses ``run_eval()`` and
        # drives ``take_action()`` directly, so without this call
        # ``RewardManager.check_list`` stays empty and ``get_reward()``
        # reports reward 1.0 for every episode (false 100% success).
        run_reward = getattr(self._env, 'run_reward', None)
        if run_reward is not None:
            run_reward()
        register_score = getattr(self._env, 'get_score', None)
        if callable(register_score):
            register_score()
        observation = robodojo_obs_to_flat(self._env.get_obs())
        info = {
            'task_id': task.task_id,
            'episode_index': int(episode_index),
            'environment_seed': environment_seed,
            'environment_case_index': layout_index,
            'layout_index': layout_index,
        }
        if self.save_videos and self._video_dir is not None:
            info['video_dir'] = str(self._video_dir)
        return observation, info

    def step(self, action: Any) -> StepResult:
        self._require_episode()
        action_dict = self._action_to_dict(action)
        try:
            self._env.take_action(action_dict)
            observation = robodojo_obs_to_flat(self._env.get_obs())
        except Exception as exc:
            if not _is_recoverable_robodojo_case_error(exc):
                raise
            raise EnvironmentCaseRejected(
                f'RoboDojo layout {self._episode_layout_index} became '
                'unstable') from exc
        terminated = bool(self._env.end_flag[0])
        success = bool(self._env.success[0]) if terminated else None
        score = self._current_score(success)
        video_paths: list[str] = []
        if terminated:
            tag = 'success' if success else 'fail'
            video_paths = self._finalize_video(self._episode_task,
                                               self._episode_index, tag)
            self._video_finalized = True
        info = {
            'task_id': self._task.task_id,
            'step': int(self._env.take_action_cnt[0]),
            'score': score,
            'environment_case_index': self._episode_layout_index,
            'layout_index': self._episode_layout_index,
        }
        if video_paths:
            info['video_paths'] = video_paths
        return StepResult(
            observation=observation,
            reward=float(bool(success)) if terminated else 0.0,
            terminated=terminated,
            truncated=False,
            success=success,
            score=score,
            info=info,
        )

    def _current_score(self, success: bool | None) -> float:
        if success:
            return 1.0
        if not callable(getattr(self._env, 'get_score', None)):
            return 0.0
        reward_manager = getattr(self._env, 'reward_manager', None)
        get_score = getattr(reward_manager, 'get_score', None)
        if not callable(get_score):
            raise RuntimeError(
                'RoboDojo task exposes get_score() without a compatible '
                'reward_manager.get_score()')
        try:
            score = float(get_score()[0]) / 100.0
        except (IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(
                'RoboDojo reward manager returned an invalid score') from exc
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(
                f'RoboDojo episode score must be in [0, 1], got {score!r}')
        return score

    def close(self) -> None:
        if self._episode_active and not self._video_finalized:
            self._finalize_video(self._episode_task, self._episode_index,
                                 'truncated')
        env = self._env
        if env is not None:
            try:
                env.close()
            finally:
                self._env = None
        simulation_app = self._simulation_app
        if simulation_app is not None:
            try:
                simulation_app.close()
            finally:
                self._simulation_app = None
        self._episode_active = False
        if self._original_cwd is not None:
            try:
                os.chdir(self._original_cwd)
            except OSError:
                pass
            self._original_cwd = None
        if self._physx_monitor is not None:
            self._physx_monitor.shutdown()
            self._physx_monitor = None
        self._video_finalized = True

    def _finalize_video(self, task: TaskSpec | None, episode_index: int | None,
                        tag: str) -> list[str]:
        """Finalize RoboDojo's streaming mp4 writers for the active episode."""
        self._video_finalized = True
        if (not self.save_videos or self._env is None
                or self._video_dir is None):
            return []
        if task is None or episode_index is None:
            return []
        save_video = getattr(self._env, 'save_video', None)
        if save_video is None:
            return []
        # RoboDojo's ``save_video`` uses ``os.replace`` to move the temporary
        # mp4 out of its local ``_stream`` directory. ``os.replace`` fails
        # across filesystems, so finalize into the env's own (local) save dir
        # first and then move the produced files to the configured video dir.
        worker_suffix = ''
        if self.worker_id is not None:
            worker_suffix = f'_worker_{self.worker_id}'
        base_name = (f'{task.task_id}_episode_{episode_index:07d}'
                     f'{worker_suffix}.mp4')
        native_save_dir = Path(self._env.save_dir)
        if not native_save_dir.is_absolute():
            native_save_dir = Path(self.robodojo_root) / native_save_dir
        native_save_dir = native_save_dir.resolve()
        local_base = native_save_dir / base_name
        captured_output = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured_output):
                save_video(0, str(local_base), tag)
            produced = sorted(native_save_dir.glob(f'{local_base.stem}_*.mp4'))
            final_paths = []
            path_replacements = {}
            for source in produced:
                target = self._video_dir / source.name
                shutil.move(str(source), str(target))
                path_replacements[str(source)] = str(target)
                final_paths.append(str(target))
        except Exception:
            return []
        output = captured_output.getvalue()
        for source, target in path_replacements.items():
            output = output.replace(source, target)
        if output:
            print(output, end='', flush=True)
        return final_paths

    def _configure_run_identity(self) -> None:
        output_run_id = self.run_id
        if output_run_id is None:
            output_run_id = os.environ.get('ROBODOJO_RUN_ID')
        if not output_run_id:
            output_run_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        native_run_id = output_run_id
        if self.worker_id is not None:
            native_run_id = f'{output_run_id}_worker_{self.worker_id}'
        os.environ['ROBODOJO_RUN_ID'] = native_run_id

        if self.save_videos and self._video_dir is not None:
            self._video_dir = self._video_dir / output_run_id

    def _action_to_dict(self, action: Any) -> dict[str, np.ndarray]:
        self._require_setup()
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.ndim != 1 or action_array.size == 0:
            raise ValueError(
                'RoboDojo action must be a one-dimensional nonempty array')
        if not np.all(np.isfinite(action_array)):
            raise ValueError('RoboDojo action must contain only finite values')

        info = self._env.robot_action_dim_info
        arm_dims = [int(value) for value in info['arm_dim']]
        ee_dims = [int(value) for value in info['ee_dim']]
        if len(arm_dims) != len(ee_dims) or len(arm_dims) not in (1, 2):
            raise ValueError(
                'RoboDojo robot_action_dim_info must describe one or two '
                'arms with matching ee_dim entries')
        arm_count = len(arm_dims)
        expected_size = self._action_size(arm_dims, ee_dims)
        if action_array.size != expected_size:
            raise ValueError(
                f'RoboDojo action_mode={self.action_mode!r} expects '
                f'{expected_size} values for {arm_count} arm(s), got '
                f'{action_array.size}')

        action_dict = {}
        offset = 0
        for arm_index in range(arm_count):
            if arm_count == 1:
                arm_key = 'arm_joint_state'
                ee_key = 'ee_joint_state'
                ee_pose_key = 'ee_pose'
            else:
                prefix = 'left_' if arm_index == 0 else 'right_'
                arm_key = f'{prefix}arm_joint_state'
                ee_key = f'{prefix}ee_joint_state'
                ee_pose_key = f'{prefix}ee_pose'
            if self.action_mode == 'joint':
                action_dict[arm_key] = action_array[offset:offset +
                                                    arm_dims[arm_index]]
                offset += arm_dims[arm_index]
            else:
                action_dict[ee_pose_key] = action_array[offset:offset +
                                                        ROBODOJO_EE_POSE_DIM]
                offset += ROBODOJO_EE_POSE_DIM
            action_dict[ee_key] = action_array[offset:offset +
                                               ee_dims[arm_index]]
            offset += ee_dims[arm_index]
        return action_dict

    def _action_size(self, arm_dims: Sequence[int],
                     ee_dims: Sequence[int]) -> int:
        arm_count = len(arm_dims)
        ee_size = int(sum(ee_dims))
        if self.action_mode == 'ee':
            return int(arm_count * ROBODOJO_EE_POSE_DIM + ee_size)
        return int(sum(arm_dims) + ee_size)

    def configure_simulator_device(self, device_id: str) -> None:
        try:
            parsed_device_id = int(device_id)
        except ValueError as exc:
            raise ValueError(
                'RoboDojo simulator GPU ids must be integer device indices, '
                f'got {device_id!r}') from exc
        self.device_id = self._nonnegative_integer(parsed_device_id,
                                                   'device_id')

    def configure_worker(self, worker_id: str, run_id: str) -> None:
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError('worker id must be a non-empty string')
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError('run id must be a non-empty string')
        self.worker_id = worker_id.strip()
        self.run_id = run_id.strip()

    def configure_video_recording(self, enabled: bool,
                                  output_dir: str | None) -> None:
        if not isinstance(enabled, bool):
            raise TypeError('enabled must be a boolean')
        if output_dir is not None:
            if not isinstance(output_dir, (str, os.PathLike)):
                raise TypeError('output_dir must be a path or None')
            output_path = Path(output_dir).expanduser()
            if not output_path.is_absolute():
                output_path = output_path.resolve()
            self._video_dir = output_path
        self.save_videos = enabled

    def _build_app_args(self, app_launcher_cls: Any) -> Any:
        parser = argparse.ArgumentParser(add_help=False)
        app_launcher_cls.add_app_launcher_args(parser)
        try:
            args = parser.parse_args([])
        except SystemExit:  # pragma: no cover - defensive
            args = argparse.Namespace()
        args.headless = self.headless
        args.enable_cameras = True
        args.video = False
        args.num_envs = self.num_envs
        args.device = f'cuda:{self._device_id}'
        args.seed = self.seed
        return args

    def _build_env_cfg(self, runtime: _PrelaunchRuntime,
                       post: _PostlaunchRuntime) -> Any:
        eval_cfg = dict(
            runtime.load_yaml(
                os.path.join(runtime.ENV_CONFIG_PATH,
                             f'{self.env_cfg_type}.yml')))
        eval_cfg['task_name'] = self._task.task_id
        eval_cfg['num_envs'] = self.num_envs
        eval_cfg['device_id'] = self._device_id
        eval_cfg['eval_batch'] = False
        eval_cfg['policy_name'] = 'fluxthemis'
        eval_cfg['additional_info'] = 'fluxthemis'
        eval_cfg['seed'] = self.seed
        eval_cfg['physx_monitor_enabled'] = True

        config = eval_cfg['config']
        config_dir = os.path.join(runtime.ROOT_DIR, 'task', runtime.BENCHMARK,
                                  'config')
        task_config = runtime.load_yaml(
            post.task_registry.task_config_path(config_dir,
                                                self._task.task_id))

        run_id = os.environ['ROBODOJO_RUN_ID']
        deploy_cfg = {
            'policy_name': 'fluxthemis',
            'port': 0,
            'host': 'localhost',
            'protocol': 'ws',
            'policy_server_url': 'ws://localhost:0',
            'evaluation_id': run_id,
            'trial_id': f'{self._task.task_id}-{run_id}',
            'action_case_id': f'{self._task.task_id}_case',
            'repeat_index': None,
        }
        env_cfg = runtime.OmegaConf.create({
            'sim':
            runtime.load_yaml(
                os.path.join(runtime.ENV_CONFIG_PATH, 'sim',
                             f"{config['sim']}.yml")),
            'scene':
            runtime.load_yaml(
                os.path.join(runtime.ENV_CONFIG_PATH, 'scene',
                             f"{config['scene']}.yml")),
            'camera':
            runtime.load_yaml(
                os.path.join(runtime.ENV_CONFIG_PATH, 'camera',
                             f"{config['camera']}.yml")),
            'robot':
            runtime.load_yaml(
                os.path.join(runtime.ENV_CONFIG_PATH, 'robot',
                             f"{config['robot']}.yml")),
            'task_env':
            task_config,
            'eval_cfg':
            eval_cfg,
            'deploy_cfg':
            deploy_cfg,
        })
        env_cfg = runtime.process_randomization(env_cfg)
        env_cfg, _ = runtime.process_config(
            env_cfg, task_name=self._task.task_id)
        runtime.OmegaConf.update(
            env_cfg, 'sim.scene.num_envs', self.num_envs, force_add=True)
        runtime.OmegaConf.update(
            env_cfg, 'eval_cfg.num_envs', self.num_envs, force_add=True)
        runtime.OmegaConf.update(
            env_cfg,
            'camera.default_frequency',
            eval_cfg['observation'].get('collect_freq', 0),
            force_add=True,
        )
        runtime.OmegaConf.update(
            env_cfg, 'sim.seed', [0] * self.num_envs, force_add=True)
        return env_cfg

    @property
    def _device_id(self) -> int:
        configured = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if configured and configured != '-1':
            try:
                visible_devices = tuple(
                    int(value.strip()) for value in configured.split(',')
                    if value.strip())
            except ValueError:
                visible_devices = ()

            # ``CUDA_VISIBLE_DEVICES`` renumbers the visible physical GPUs
            # as cuda:0..cuda:N-1 inside this process. A single visible GPU
            # (the one-device worker case) is therefore always cuda:0,
            # regardless of its physical id. With multiple visible GPUs,
            # map the configured physical ``device_id`` to its position in
            # the visible-device list.
            if len(visible_devices) == 1:
                return 0
            if self.device_id is not None:
                try:
                    return visible_devices.index(self.device_id)
                except ValueError:
                    return 0
            return 0

        if self.device_id is not None:
            return self.device_id
        return 0

    def _prepare_runtime_paths(self) -> None:
        root = Path(self.robodojo_root)
        if not root.is_dir():
            raise FileNotFoundError(
                f'RoboDojo checkout not found at {root}; set robodojo_root '
                'or the ROBODOJO_ROOT environment variable')
        missing = [
            subdir for subdir in ROBODOJO_REQUIRED_SUBDIRS
            if not (root / subdir).is_dir()
        ]
        if missing:
            raise FileNotFoundError(
                f'RoboDojo checkout at {root} is incomplete; missing '
                f'{", ".join(missing)}. Run '
                '`bash scripts/install_robodojo.sh`.')
        self._original_cwd = os.getcwd()
        self._sync_runtime_paths()

    def _sync_runtime_paths(self) -> None:
        """Make RoboDojo's root the first entry in ``sys.path``.

        Both the checkout root and ``XPolicyLab`` contain a top-level
        ``utils`` namespace package. RoboDojo simulator modules use
        ``from utils.load_file import *``, which only works when the root
        copy is found first. Isaac Sim can mutate ``sys.path`` during
        startup, so this is deliberately cheap to call more than once.
        """
        root = Path(self.robodojo_root)
        # Insert XPolicyLab first and the RoboDojo root second, so the root
        # ends up at sys.path[0]. Both directories contain a top-level
        # ``utils`` namespace package, but RoboDojo's simulator code needs
        # the root version (``load_object_metadata`` et al.); XPolicyLab's
        # copy would otherwise shadow it.
        for path in (str(root / 'XPolicyLab'), str(root)):
            while path in sys.path:
                sys.path.remove(path)
            sys.path.insert(0, path)
        os.chdir(root)

    def _require_setup(self) -> None:
        if self._env is None:
            raise RuntimeError('RoboDojoEnvironment.setup() was not called')

    def _require_episode(self) -> None:
        self._require_setup()
        if not self._episode_active:
            raise RuntimeError(
                'RoboDojoEnvironment.reset() must be called before step()')

    @staticmethod
    def _nonnegative_integer(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f'{name} must be an integer')
        value = int(value)
        if value < 0:
            raise ValueError(f'{name} must be non-negative')
        return value


class _NoOpModelClient:
    """Stand-in for the XPolicyLab WebSocket client when FluxThemis drives
    the environment directly."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def call(self, func_name: str, **kwargs: Any) -> dict[str, Any]:
        return {'status': 'ok', 'func_name': func_name}

    def close(self) -> None:
        pass


def _read_env_cfg_config(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError:
        return {}
    try:
        value = yaml.safe_load(path.read_text(encoding='utf-8'))
    except Exception:
        return {}
    return value if isinstance(value, Mapping) else {}
