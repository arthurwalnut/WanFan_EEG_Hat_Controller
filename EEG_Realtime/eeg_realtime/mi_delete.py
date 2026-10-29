"""Two-step deletion of explicitly previewed, inactive MI recording folders.

The engine serializes calls and excludes capture, training and validation. No
recording is removed by preview; confirmation is bound to a short-lived snapshot.
"""
from __future__ import annotations

from pathlib import Path
import secrets
import stat
import time

from .mi_recording import _read_manifest


TTL_SECONDS = 300
MAX_SELECTION = 500
MAX_ENTRIES = 10000
MAX_MANIFEST_BYTES = 2_000_000


def _fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _linked(info):
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


class RecordingDeletion:
    def __init__(self, root):
        self.root = Path(root).absolute()
        self._root_resolved = self.root.resolve()
        self._previews = {}

    def _check_root(self):
        info = self.root.lstat()
        if (_linked(info) or not stat.S_ISDIR(info.st_mode)
                or self.root.resolve() != self._root_resolved):
            raise ValueError("录制根目录已改变或包含链接，请重新检查数据位置。")

    @staticmethod
    def _ids(identifiers):
        if (not isinstance(identifiers, list) or not identifiers
                or len(identifiers) > MAX_SELECTION):
            raise ValueError("请选择 1–500 个录制轮次。")
        for identifier in identifiers:
            if (not isinstance(identifier, str) or not identifier
                    or identifier in (".", "..")
                    or any(c in identifier for c in '/\\:\x00')
                    or Path(identifier).name != identifier
                    or identifier.endswith((" ", "."))):
                raise ValueError("只能删除录制列表中的目录，不能使用路径。")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("删除列表中有重复的录制轮次。")
        return tuple(identifiers)

    def _safe_entry(self, path, *, directory):
        self._check_root()
        # Recheck all ancestors before each destructive operation; a directory
        # replaced by a Windows junction must not redirect the target elsewhere.
        relative = path.relative_to(self.root)
        if not relative.parts:
            raise ValueError("不能删除录制根目录。")
        current = self.root
        for part in relative.parts:
            current /= part
            info = current.lstat()
            if _linked(info):
                raise ValueError(f"录制包含链接或重解析点：{relative.as_posix()}")
            if current != path and not stat.S_ISDIR(info.st_mode):
                raise ValueError("录制目录结构已改变，请重新预览。")
        if not path.resolve().is_relative_to(self._root_resolved):
            raise ValueError("录制路径超出数据目录。")
        expected = stat.S_ISDIR if directory else stat.S_ISREG
        if not expected(info.st_mode):
            raise ValueError(f"录制文件类型不受支持：{relative.as_posix()}")
        return info

    def _snapshot(self, identifier):
        folder = self.root / identifier
        directory_info = self._safe_entry(folder, directory=True)
        entries = {"": ("directory", _fingerprint(directory_info))}
        queue = [folder]
        while queue:
            directory = queue.pop()
            self._safe_entry(directory, directory=True)
            for path in sorted(directory.iterdir()):
                info = path.lstat()
                is_directory = stat.S_ISDIR(info.st_mode)
                info = self._safe_entry(path, directory=is_directory)
                relative = path.relative_to(folder).as_posix()
                entries[relative] = ("directory" if is_directory else "file", _fingerprint(info))
                if len(entries) > MAX_ENTRIES:
                    raise ValueError("录制目录条目过多，不能通过页面删除。")
                if is_directory:
                    queue.append(path)
        manifest = folder / "manifest.json"
        manifest_info = self._safe_entry(manifest, directory=False)
        if manifest_info.st_size > MAX_MANIFEST_BYTES:
            raise ValueError("录制清单过大，不能通过页面删除。")
        metadata = _read_manifest(folder)
        if metadata["id"] != identifier:
            raise ValueError("录制编号与目录不一致，不能通过页面删除。")
        if _fingerprint(self._safe_entry(manifest, directory=False)) != entries["manifest.json"][1]:
            raise ValueError("录制清单已改变，请重新预览。")
        return {"id": identifier, "subject": metadata.get("subject", ""), "entries": entries}

    def _prune(self):
        now = time.monotonic()
        self._previews = {key: item for key, item in self._previews.items() if item["expires"] > now}

    def preview(self, identifiers):
        identifiers = self._ids(identifiers)
        self._prune()
        snapshots, total_entries = [], 0
        for identifier in identifiers:
            snapshot = self._snapshot(identifier)
            total_entries += len(snapshot["entries"])
            if total_entries > MAX_ENTRIES:
                raise ValueError("所选目录条目过多，请分批删除。")
            snapshots.append(snapshot)
        token = secrets.token_urlsafe(24)
        # Bound memory if a client opens many confirmation dialogs.
        if len(self._previews) >= 8:
            self._previews.pop(next(iter(self._previews)))
        self._previews[token] = {"expires": time.monotonic() + TTL_SECONDS, "snapshots": snapshots}
        return {"ok": True, "token": token, "expires_in_s": TTL_SECONDS,
                "recordings": [{"id": item["id"], "subject": item["subject"],
                                "files": sorted(name for name, entry in item["entries"].items()
                                                if entry[0] == "file")}
                               for item in snapshots]}

    def _preview(self, token):
        self._prune()
        if not isinstance(token, str) or token not in self._previews:
            raise ValueError("删除确认已过期或无效，请重新选择并预览。")
        return self._previews[token]

    def selected(self, token):
        return [item["id"] for item in self._preview(token)["snapshots"]]

    def delete(self, token):
        preview = self._preview(token)
        # Consume once, including when validation fails. A new preview is needed
        # after a partial deletion, a changed file, or an operating-system error.
        del self._previews[token]
        snapshots = preview["snapshots"]
        for original in snapshots:
            if self._snapshot(original["id"]) != original:
                raise ValueError(f"录制 {original['id']} 的文件已改变，请重新预览后确认。")
        deleted, errors = [], []
        for item in snapshots:
            folder = self.root / item["id"]
            try:
                # Check again after earlier selected directories were removed.
                if self._snapshot(item["id"]) != item:
                    raise ValueError("文件已改变，请重新预览。")
                for relative, (kind, fingerprint) in sorted(item["entries"].items()):
                    if kind != "file" or relative == "manifest.json":
                        continue
                    path = folder / relative
                    info = self._safe_entry(path, directory=False)
                    if _fingerprint(info) != fingerprint:
                        raise ValueError("文件已改变，请重新预览。")
                    path.unlink()
                directories = [name for name, entry in item["entries"].items()
                               if name and entry[0] == "directory"]
                for relative in sorted(directories, key=lambda name: len(Path(name).parts), reverse=True):
                    path = folder / relative
                    self._safe_entry(path, directory=True)
                    # rmdir refuses newly added files, unlike recursive removal.
                    path.rmdir()
                # Keep the manifest/list entry until all other files and child
                # directories have been removed successfully.
                self._safe_entry(folder, directory=True)
                manifest = folder / "manifest.json"
                if list(folder.iterdir()) != [manifest]:
                    raise ValueError("目录中出现未确认的文件，请重新预览。")
                info = self._safe_entry(manifest, directory=False)
                if _fingerprint(info) != item["entries"]["manifest.json"][1]:
                    raise ValueError("录制清单已改变，请重新预览。")
                manifest.unlink()
                self._safe_entry(folder, directory=True)
                folder.rmdir()
                deleted.append(item["id"])
            except (OSError, ValueError) as exc:
                errors.append(f"{item['id']}：{exc}；该目录可能已删除部分文件，请刷新列表检查。")
                break
        pending = [item["id"] for item in snapshots if item["id"] not in deleted]
        return {"ok": not errors, "deleted": deleted, "errors": errors, "remaining": pending}
