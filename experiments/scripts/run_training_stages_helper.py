"""
This script is a helper to run DVC stages in parallel using tmux.
You don't have to use it to reproduce the results, you can use the DVC commands directly.
"""

import json
import subprocess
from itertools import cycle
from time import sleep
from typing import Tuple

import typer
from libtmux.pane import Pane
from libtmux.server import Server
from libtmux.session import Session
from mpire import WorkerPool
from rich import print

server = Server()
print(server)


def work(worker_id: int, shared_objects: Tuple[Session, dict[int, str] | None], stage: str) -> None:
    sleep(worker_id * 10)
    session, worker_device_assignments = shared_objects
    if worker_device_assignments is not None:
        cmd = ["CUDA_VISIBLE_DEVICES=" + worker_device_assignments[worker_id]]
    else:
        cmd = []
    cmd += ["poetry", "run", "dvc", "repro", "-s", stage]
    stage_name = stage.split(":")[-1]
    print(f'Starting "{stage}" on worker {worker_id}: \n\t{" ".join(cmd)}')
    window = session.new_window(
        attach=True, window_name=f"w{worker_id}:{stage_name}", window_shell="bash"
    )
    assert isinstance(window.attached_pane, Pane)
    print("\tSending cmd...")
    window.attached_pane.send_keys(
        " ".join(cmd) + f"; tmux wait -S ping-{session.session_name}-{worker_id}"
    )
    print(f"\tWaiting for ping-{session.session_name}-{worker_id}...")
    subprocess.run(["tmux", "wait", f"ping-{session.session_name}-{worker_id}"]).check_returncode()
    print(f'Finished "{stage}"...')


def main(
    targets: str = typer.Argument(..., help="DVC targets to run e.g. dvc.yaml:training"),
    n_jobs: int = typer.Option(...),
    session_name: str = typer.Option("mgi_runner"),
    device: list[str] = typer.Option(None),
) -> None:
    try:
        session = server.sessions.filter(session_name=session_name)[0]
    except IndexError:
        session = server.new_session(session_name=session_name)

    if device is not None and len(device) > 0:
        worker_device_assignments = {}
        for cuda_device, worker_id in zip(cycle(device), range(n_jobs)):
            worker_device_assignments[worker_id] = cuda_device
    else:
        worker_device_assignments = None

    result = subprocess.run(["dvc", "status", "--json", targets], stdout=subprocess.PIPE)
    result.check_returncode()

    stages = sorted(json.loads(result.stdout.decode().replace("\x1b[0m", "")).keys())
    shared_objects = (session, worker_device_assignments)
    with WorkerPool(n_jobs=n_jobs, pass_worker_id=True, shared_objects=shared_objects) as pool:
        pool.map(work, stages, progress_bar=True)


typer.run(main)
