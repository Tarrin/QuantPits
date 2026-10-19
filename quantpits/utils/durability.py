"""Filesystem durability helpers shared by training and runtime writers."""

from __future__ import annotations

import os
from pathlib import Path

try:
    import fcntl

    flock_compat = fcntl
except ImportError:  # pragma: no cover - production is Linux
    import msvcrt

    class _MsvcrtFlockShim:
        """Minimal fcntl.flock replacement over msvcrt byte-range locks.

        The lock always covers one byte at offset 0, which is valid on Windows
        even for empty files.  The seek is issued on the raw descriptor right
        before locking/unlocking so the locked range is position-independent
        for callers.  Unlike advisory POSIX locks, Windows byte-range locks
        are mandatory: other handles (including read handles) are denied until
        the lock is released.
        """

        LOCK_SH = 0x1
        LOCK_EX = 0x2
        LOCK_NB = 0x4
        LOCK_UN = 0x8

        @staticmethod
        def flock(fd, flags):
            os.lseek(fd, 0, os.SEEK_SET)
            if flags & _MsvcrtFlockShim.LOCK_UN:
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                return
            # msvcrt only offers exclusive byte-range locks; a shared request
            # is conservatively mapped onto an exclusive lock.
            mode = msvcrt.LK_NBLCK if flags & _MsvcrtFlockShim.LOCK_NB else msvcrt.LK_LOCK
            msvcrt.locking(fd, mode, 1)

    flock_compat = _MsvcrtFlockShim


def fsync_file(path: Path | str) -> None:
    """Flush a regular file's contents to stable storage.

    Windows rejects fsync on read-only descriptors (EBADF), so the handle is
    opened read-write there; POSIX keeps O_RDONLY.
    """

    flags = os.O_RDWR if os.name == "nt" else os.O_RDONLY
    descriptor = os.open(str(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: Path | str) -> None:
    """Flush a directory entry so a just-replaced file survives power loss.

    Opening directories is a POSIX capability: Linux exposes O_RDONLY
    directory handles for fsync, while Windows cannot open directories at
    all (Errno 13).  NTFS metadata is journaled, so the POSIX-only directory
    fsync is simply skipped on Windows.
    """

    if os.name == "nt":
        return
    directory_fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


__all__ = ["flock_compat", "fsync_directory", "fsync_file"]
