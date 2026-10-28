"""Bounded native OS locking, two-process patches and atomic reader controls."""

import multiprocessing
import os
from pathlib import Path

import pytest
from src.overrides import store as storage
from test_override_store import _open_store

pytestmark = pytest.mark.timeout(90)


def _native_holder(path, channel):
    with open(str(path) + ".lock", "r+b") as handle:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        channel.send(("locked", os.name))
        assert channel.recv() == "release"
    channel.close()


def _patch_writer(backend, path, override_id, channel, gated):
    store = _open_store(backend, Path(path))
    original = storage._write_override_frame_atomically

    def gated_write(frame, destination):
        channel.send("read-and-locked")
        assert channel.recv() == "release"
        original(frame, destination)

    try:
        if gated:
            storage._write_override_frame_atomically = gated_write
            assert store.update_override(override_id, {"notes": "first writer"})
        else:
            try:
                store.update_override(override_id, {"proportion": 0.0})
            except storage.OverrideStoreBusy:
                channel.send("busy")
            else:
                raise AssertionError("The held native writer lock was not enforced")
            assert channel.recv() == "retry"
            assert store.update_override(override_id, {"proportion": 0.0})
        channel.send("committed")
    finally:
        storage._write_override_frame_atomically = original
        store.close()
        channel.close()


def _repeated_writer(backend, path, override_id, channel):
    store = _open_store(backend, Path(path))
    try:
        channel.send("ready")
        assert channel.recv() == "start"
        for index in range(12):
            try:
                assert store.update_override(override_id, {"notes": f"complete-{index}"})
            except storage.OverrideStoreBusy:
                channel.send("busy")
                assert channel.recv() == "retry"
                assert store.update_override(override_id, {"notes": f"complete-{index}"})
        channel.send("committed")
    finally:
        store.close()
        channel.close()


def _start(target, *args):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=target, args=(*args, child))
    process.start()
    child.close()
    return process, parent


def _receive(channel):
    assert channel.poll(30), "Override process exceeded its bounded message deadline"
    return channel.recv()


def _finish(process):
    process.join(timeout=20)
    assert not process.is_alive(), "Override process did not finish"
    assert process.exitcode == 0


def _stop(process, channel):
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)
    channel.close()
    process.close()


@pytest.mark.parametrize("backend", ["parquet", "csv"])
@pytest.mark.parametrize("operation", ["add", "update", "delete"])
def test_native_os_lock_blocks_every_mutation_without_touching_old_bytes(
    tmp_path, backend, operation, record_property,
):
    path = tmp_path / f"overrides.{backend}"
    store = _open_store(backend, path)
    override_id = store.add_override("old", "new", 0.75, notes="before")
    before = path.read_bytes()
    inode = path.with_name(path.name + ".lock").stat().st_ino
    process, channel = _start(_native_holder, path)
    try:
        assert _receive(channel) == ("locked", os.name)
        record_property("native_lock_api", "msvcrt.locking" if os.name == "nt" else "fcntl.flock")
        with pytest.raises(storage.OverrideStoreBusy):
            if operation == "add":
                store.add_override("other", "newer", 0.5)
            elif operation == "update":
                store.update_override(override_id, {"notes": "after"})
            else:
                store.delete_override(override_id)
        assert path.read_bytes() == before
        with pytest.raises(storage.OverrideStoreBusy):
            store.get_all()
        independent = _open_store(backend, tmp_path / f"independent.{backend}")
        try:
            assert independent.add_override("a", "b", 0.25)
        finally:
            independent.close()
        channel.send("release")
        _finish(process)
        assert store.update_override(override_id, {"notes": "explicit retry"})
        assert path.with_name(path.name + ".lock").stat().st_ino == inode
    finally:
        store.close()
        _stop(process, channel)


@pytest.mark.parametrize("backend", ["parquet", "csv"])
def test_two_process_independent_field_patches_survive_explicit_busy_retry(tmp_path, backend):
    path = tmp_path / f"overrides.{backend}"
    store = _open_store(backend, path)
    override_id = store.add_override("old", "new", 0.75, notes="before")
    other_id = store.add_override("other", "newer", 0.5, notes="unchanged")
    before = {row["override_id"]: row for row in store.get_all().to_dicts()}
    before_bytes = path.read_bytes()
    context = multiprocessing.get_context("spawn")
    first_channel, first_child = context.Pipe()
    second_channel, second_child = context.Pipe()
    first = context.Process(target=_patch_writer, args=(backend, path, override_id, first_child, True))
    second = context.Process(target=_patch_writer, args=(backend, path, override_id, second_child, False))
    first.start()
    first_child.close()
    second_started = False
    try:
        assert _receive(first_channel) == "read-and-locked"
        second.start()
        second_started = True
        second_child.close()
        assert _receive(second_channel) == "busy"
        with pytest.raises(storage.OverrideStoreBusy):
            store.get_all()
        assert path.read_bytes() == before_bytes
        first_channel.send("release")
        assert _receive(first_channel) == "committed"
        _finish(first)
        second_channel.send("retry")
        assert _receive(second_channel) == "committed"
        _finish(second)
        reopened = _open_store(backend, path)
        try:
            after = {row["override_id"]: row for row in reopened.get_all().to_dicts()}
            assert after[override_id] == {**before[override_id], "notes": "first writer", "proportion": 0.0}
            assert after[other_id] == before[other_id]
        finally:
            reopened.close()
    finally:
        store.close()
        _stop(first, first_channel)
        if second_started:
            _stop(second, second_channel)
        else:
            second_child.close()
            second_channel.close()
            second.close()


@pytest.mark.parametrize("backend", ["parquet", "csv"])
def test_readers_observe_complete_old_or_new_files_during_replacement(tmp_path, backend):
    path = tmp_path / f"overrides.{backend}"
    store = _open_store(backend, path)
    override_id = store.add_override("old", "new", 0.75, notes="before")
    store.add_override("other", "newer", 0.5, notes="unchanged")
    process, channel = _start(_repeated_writer, backend, path, override_id)
    try:
        assert _receive(channel) == "ready"
        channel.send("start")
        for _ in range(40):
            try:
                frame = store.get_all()
            except storage.OverrideStoreBusy:
                continue
            assert frame.height == 2
            observed = {row["override_id"]: row for row in frame.to_dicts()}
            assert observed[override_id]["notes"] in {"before", *(f"complete-{i}" for i in range(12))}
            assert observed[override_id]["proportion"] == 0.75
        for _ in range(13):
            result = _receive(channel)
            if result == "committed":
                break
            assert result == "busy"
            channel.send("retry")
        assert result == "committed"
        _finish(process)
        assert store.get_all().filter(storage.pl.col("override_id") == override_id)["notes"][0] == "complete-11"
    finally:
        store.close()
        _stop(process, channel)
