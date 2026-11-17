import concurrent.futures
import io
import json
import multiprocessing
import os
import signal
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from sdr_harvest.cli import (
    RESOURCE_TRACKER_WARNING_FILTER,
    _configure_child_warning_filters,
    _interrupt_on_sigterm,
    _settings,
    main,
    parser,
)
from sdr_harvest.core import (
    StageError,
    exit_with_parent,
    interruptible_process_pool,
    interruptible_thread_pool,
)
from sdr_harvest.state import StateStore
from tests.support import DRUID


class InterruptTest(unittest.TestCase):
    def test_resource_tracker_filter_is_targeted_preserved_and_idempotent(self):
        existing = "default::DeprecationWarning"
        with patch.dict(os.environ, {"PYTHONWARNINGS": existing}):
            _configure_child_warning_filters()
            _configure_child_warning_filters()
            filters = os.environ["PYTHONWARNINGS"].split(",")
            self.assertIn(existing, filters)
            self.assertEqual(1, filters.count(RESOURCE_TRACKER_WARNING_FILTER))

    def test_thread_pool_does_not_wait_after_keyboard_interrupt(self):
        executor = Mock(spec=ThreadPoolExecutor)
        with patch(
            "sdr_harvest.pipeline.concurrent.futures.ThreadPoolExecutor",
            return_value=executor,
        ):
            with self.assertRaises(KeyboardInterrupt):
                with interruptible_thread_pool(4):
                    raise KeyboardInterrupt
        executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)

    def test_process_pool_terminates_busy_workers_after_keyboard_interrupt(self):
        with self.assertRaises(KeyboardInterrupt):
            with interruptible_process_pool(1) as executor:
                future = executor.submit(time.sleep, 60)
                deadline = time.monotonic() + 30
                while not future.running() and time.monotonic() < deadline:
                    time.sleep(0.05)
                processes = list(executor._processes.values())
                raise KeyboardInterrupt
        for process in processes:
            process.join(timeout=10)
            self.assertFalse(process.is_alive())

    def test_process_pool_failure_terminates_running_workers(self):
        with self.assertRaises(RuntimeError):
            with interruptible_process_pool(1) as executor:
                executor.submit(time.sleep, 60)
                processes = list(executor._processes.values())
                raise RuntimeError("run failed")
        for process in processes:
            process.join(timeout=10)
            self.assertFalse(process.is_alive())

    def test_pool_worker_exits_once_its_parent_is_gone(self):
        # A parent PID that is not the worker's parent stands in for a
        # coordinator that was killed without cleaning up its pool.
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=exit_with_parent,
            initargs=(-1, 0.05),
        ) as executor:
            future = executor.submit(time.sleep, 60)
            with self.assertRaises(concurrent.futures.process.BrokenProcessPool):
                future.result(timeout=30)

    def test_sigterm_interrupts_like_ctrl_c(self):
        previous = signal.signal(signal.SIGTERM, _interrupt_on_sigterm)
        try:
            with self.assertRaises(KeyboardInterrupt):
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(5)
        finally:
            signal.signal(signal.SIGTERM, previous)

    def test_cli_exits_immediately_with_status_130(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            manifest.write_text(f"identifier\n{DRUID}\n")
            with (
                patch("sdr_harvest.cli.Pipeline.run", side_effect=KeyboardInterrupt),
                patch("sdr_harvest.cli.os._exit", side_effect=SystemExit(130)) as exit_,
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                main(
                    [
                        "--state-dir",
                        str(root / "state"),
                        "run",
                        "--manifest",
                        str(manifest),
                        "--no-progress",
                    ]
                )
            self.assertEqual(130, raised.exception.code)
            exit_.assert_called_once_with(130)


class CliErrorTest(unittest.TestCase):
    def test_parser_handler_can_run_plan_without_main(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            manifest.write_text(f"identifier\n{DRUID}\n")
            state_dir = root / "state"
            args = parser().parse_args(
                [
                    "--state-dir",
                    str(state_dir),
                    "--dark-list",
                    str(root / "dark.csv"),
                    "plan",
                    "--manifest",
                    str(manifest),
                ]
            )
            store = StateStore(state_dir / "state.sqlite3")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                args.handler(root, args, store)

            self.assertEqual("_command_plan", args.handler.__name__)
            self.assertEqual(1, json.loads(stdout.getvalue())["manifest_objects"])
            store.close()

    def test_build_can_retain_failed_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parser().parse_args(
                [
                    "run",
                    "--manifest",
                    "manifest.csv",
                    "--keep-failed-downloads",
                ]
            )
            args.state_dir = root / "state"

            self.assertTrue(_settings(root, args).keep_failed_downloads)

    def test_publish_insecure_disables_tls_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parser().parse_args(
                [
                    "publish",
                    "--manifest",
                    "manifest.csv",
                    "--target",
                    "https://localhost:8983/solr/collection",
                    "--insecure",
                ]
            )
            args.state_dir = root / "state"

            self.assertFalse(_settings(root, args).verify_tls)

    def test_stage_error_is_printed_without_an_exception_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            manifest.write_text(f"identifier\n{DRUID}\n")
            stderr = io.StringIO()
            with (
                patch(
                    "sdr_harvest.cli.Pipeline.run",
                    side_effect=StageError("Another sdr-harvest run is already active"),
                ),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as raised,
            ):
                main(
                    [
                        "--state-dir",
                        str(root / "state"),
                        "run",
                        "--manifest",
                        str(manifest),
                        "--no-progress",
                    ]
                )

            self.assertEqual(1, raised.exception.code)
            self.assertEqual(
                "Error: Another sdr-harvest run is already active\n",
                stderr.getvalue(),
            )
