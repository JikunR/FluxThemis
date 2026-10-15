"""Restartable simulator worker and its local process supervisor."""

from __future__ import annotations

import multiprocessing
import os
import signal
import time
import traceback
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from fluxthemis.engines.utils import (build_environment_from_cfg,
                                      build_policy_client_from_cfg,
                                      to_jsonable)
from fluxthemis.environments.base import (BaseEnvironment,
                                          EnvironmentCaseRejected)
from fluxthemis.model_clients.base import BasePolicyClient
from fluxthemis.structures import TaskSpec

from ...transports.zmq_protocol import (decode_frame, decode_header,
                                        empty_payload, encode_frame,
                                        encode_header)
from .config import (distributed_video_settings,
                     load_distributed_evaluation_config)
from .episode import (EpisodeJob, configure_simulator_gpu,
                      configure_worker_lifetime, execute_episode)


class SimulatorWorkerSupervisor:
    """Restart one disposable simulator worker process when it exits."""

    def __init__(self,
                 config_path: str,
                 rollout_endpoint: str,
                 worker_id: str,
                 deployment_id: str,
                 device: str,
                 episode_timeout_s: float,
                 config_overrides: Mapping[str, Any] | None = None) -> None:
        self.config_path = config_path
        self.rollout_endpoint = _endpoint(rollout_endpoint, 'rollout_endpoint')
        self.worker_id = _nonempty(worker_id, 'worker_id')
        self.deployment_id = _nonempty(deployment_id, 'deployment_id')
        self.device = _nonempty(device, 'device')
        self.episode_timeout_s = _positive_float(episode_timeout_s,
                                                 'episode_timeout_s')
        self.config_overrides = dict(config_overrides or {})
        self._context = multiprocessing.get_context('spawn')

    def run(self) -> None:
        generation = 0
        while True:
            process = self._context.Process(
                target=_run_simulator_worker,
                kwargs={
                    'config_path': self.config_path,
                    'rollout_endpoint': self.rollout_endpoint,
                    'worker_id': self.worker_id,
                    'deployment_id': self.deployment_id,
                    'generation': generation,
                    'device': self.device,
                    'episode_timeout_s': self.episode_timeout_s,
                    'config_overrides': self.config_overrides,
                    'parent_pid': os.getpid(),
                },
                name=self.worker_id,
            )
            process.start()
            try:
                process.join()
            except BaseException:
                _terminate_process(process)
                raise
            if process.exitcode == 0:
                return
            exit_code = process.exitcode
            generation += 1
            print(
                f'[FluxThemis] restarting {self.worker_id} after exit code '
                f'{exit_code} as generation {generation}',
                flush=True,
            )
            time.sleep(0.5)


class _SimulatorWorker:

    def __init__(self, *, config_path: str, rollout_endpoint: str,
                 worker_id: str, deployment_id: str, generation: int,
                 device: str, episode_timeout_s: float,
                 config_overrides: Mapping[str, Any]) -> None:
        self.config_path = config_path
        self.rollout_endpoint = rollout_endpoint
        self.worker_id = worker_id
        self.deployment_id = deployment_id
        self.generation = generation
        self.device = device
        self.episode_timeout_s = episode_timeout_s
        self.config_overrides = config_overrides

    @property
    def _message_identity(self) -> dict[str, Any]:
        return {
            'deployment_id': self.deployment_id,
            'worker_id': self.worker_id,
            'generation': self.generation,
        }

    def run(self) -> None:
        import zmq

        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        socket.setsockopt(
            zmq.IDENTITY,
            f'{self.worker_id}:{self.generation}'.encode(),
        )
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.rollout_endpoint)
        environment: BaseEnvironment | None = None
        policy_client: BasePolicyClient | None = None
        try:
            self._register(socket)
            evaluation_cfg = load_distributed_evaluation_config(
                self.config_path, config_overrides=self.config_overrides)
            save_videos, video_dir, video_fps = distributed_video_settings(
                evaluation_cfg)
            environment = build_environment_from_cfg(
                evaluation_cfg['environment'])
            environment.configure_worker(self.worker_id, self.deployment_id)
            configure_simulator_gpu(environment, self.device)
            environment.configure_video_recording(save_videos, video_dir)
            policy_client = build_policy_client_from_cfg(
                evaluation_cfg['policy_client'])
            policy_client.setup()
            environment.setup()
            tasks = tuple(
                TaskSpec.from_value(task) for task in environment.tasks())
            socket.send_multipart([
                encode_header('ready', **self._message_identity),
                encode_frame({'tasks': [to_jsonable(task) for task in tasks]}),
            ])
            while True:
                job = self._receive_job(socket)
                if job is None:
                    return
                self._execute_job(
                    socket=socket,
                    environment=environment,
                    policy_client=policy_client,
                    evaluation_cfg=evaluation_cfg,
                    job=job,
                    save_videos=save_videos,
                    video_dir=video_dir,
                    video_fps=video_fps,
                )
        except BaseException:
            traceback.print_exc()
            os._exit(1)
        finally:
            for component in (environment, policy_client):
                if component is None:
                    continue
                try:
                    component.close()
                except Exception:
                    pass
            socket.close(linger=0)
            context.term()

    def _register(self, socket: Any) -> None:
        socket.send_multipart([
            encode_header(
                'register_worker',
                **self._message_identity,
                device=self.device,
            ),
            empty_payload(),
        ])

    def _receive_job(self, socket: Any) -> EpisodeJob | None:
        frames = socket.recv_multipart()
        header = decode_header(frames[0])
        self._validate_header(header)
        if header['type'] == 'shutdown_worker':
            return None
        if header['type'] != 'episode_job':
            raise RuntimeError(
                f'Unexpected simulator command {header["type"]!r}')
        payload = frames[1] if len(frames) > 1 else empty_payload()
        job = _episode_job_from_wire(decode_frame(payload))
        if header.get('episode_id') != job.episode_id:
            raise RuntimeError('simulator episode_id mismatch')
        return job

    def _execute_job(self, *, socket: Any, environment: BaseEnvironment,
                     policy_client: BasePolicyClient,
                     evaluation_cfg: Mapping[str, Any], job: EpisodeJob,
                     save_videos: bool, video_dir: str | None,
                     video_fps: int) -> None:
        with _episode_timeout(self.episode_timeout_s):

            def report_episode_start() -> None:
                _send_episode_event(
                    socket,
                    self._message_identity,
                    'episode_start',
                    job.episode_id,
                )

            try:
                result = execute_episode(
                    environment=environment,
                    policy_client=policy_client,
                    job=job,
                    max_episode_steps=int(evaluation_cfg['max_episode_steps']),
                    execute_horizon=evaluation_cfg.get('execute_horizon'),
                    stop_on_success=bool(
                        evaluation_cfg.get('stop_on_success', True)),
                    on_episode_started=report_episode_start,
                    save_videos=save_videos,
                    video_dir=video_dir,
                    video_fps=video_fps,
                    video_worker_id=self.worker_id,
                )
            except EnvironmentCaseRejected as exc:
                _send_episode_event(
                    socket,
                    self._message_identity,
                    'environment_case_rejected',
                    job.episode_id,
                    message=str(exc),
                )
                return
            _send_episode_event(
                socket,
                self._message_identity,
                'episode_end',
                job.episode_id,
                payload=encode_frame(to_jsonable(result)),
            )

    def _validate_header(self, header: Mapping[str, Any]) -> None:
        if header.get('deployment_id') != self.deployment_id:
            raise RuntimeError('ZMQ deployment_id mismatch')
        if header.get('worker_id') != self.worker_id:
            raise RuntimeError('simulator worker_id mismatch')
        if header.get('generation') != self.generation:
            raise RuntimeError('simulator worker generation mismatch')


def _run_simulator_worker(*, config_path: str, rollout_endpoint: str,
                          worker_id: str, deployment_id: str, generation: int,
                          device: str, episode_timeout_s: float,
                          config_overrides: Mapping[str, Any],
                          parent_pid: int) -> None:
    configure_worker_lifetime(parent_pid)
    _SimulatorWorker(
        config_path=config_path,
        rollout_endpoint=rollout_endpoint,
        worker_id=worker_id,
        deployment_id=deployment_id,
        generation=generation,
        device=device,
        episode_timeout_s=episode_timeout_s,
        config_overrides=config_overrides,
    ).run()


def _send_episode_event(socket: Any,
                        message_identity: Mapping[str, Any],
                        event_type: str,
                        episode_id: str,
                        payload: bytes | None = None,
                        **values: Any) -> None:
    socket.send_multipart([
        encode_header(
            event_type,
            **message_identity,
            episode_id=episode_id,
            **values,
        ),
        payload if payload is not None else empty_payload(),
    ])


def _episode_job_from_wire(value: Any) -> EpisodeJob:
    if not isinstance(value, Mapping):
        raise TypeError('episode job payload must be a mapping')
    data = dict(value)
    data['task'] = TaskSpec.from_value(data['task'])
    return EpisodeJob(**data)


@contextmanager
def _episode_timeout(timeout_s: float) -> Iterator[None]:
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _terminate_process(process: multiprocessing.Process) -> None:
    if process.is_alive():
        process.terminate()
        process.join(timeout=10.0)
    if process.is_alive():
        process.kill()
        process.join()


def _endpoint(value: Any, name: str) -> str:
    value = _nonempty(value, name)
    if not value.startswith(('tcp://', 'ipc://', 'inproc://')):
        raise ValueError(f'{name} must be a complete ZMQ endpoint')
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')
    return value.strip()


def _positive_float(value: Any, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or value <= 0):
        raise ValueError(f'{name} must be a positive number')
    return float(value)


__all__ = ['SimulatorWorkerSupervisor']
