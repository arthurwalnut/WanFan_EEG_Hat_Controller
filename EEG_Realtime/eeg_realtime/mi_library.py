"""Local display names, favorites and confirmed deletion for MI library items.

Names and stars live in small sidecars. Recording IDs, subject labels, raw EEG,
model bytes and training provenance remain unchanged. The engine serializes
these operations with recording, training, validation and replay activity.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import stat
import time
import unicodedata
from uuid import uuid4

from .mi_delete import RecordingDeletion, TTL_SECONDS, _fingerprint, _linked
from .mi_recording import _read_manifest


MAX_LIBRARY_BYTES = 16_384
MAX_MODEL_SUMMARY_BYTES = 1_000_000
MAX_DISPLAY_NAME = 80
_UNSET = object()


def _identifier(identifier):
    RecordingDeletion._ids([identifier])
    # Sidecar stems are reserved and cannot masquerade as separate models.
    if identifier.casefold().endswith('.library'):
        raise ValueError("请选择列表中的数据或模型编号。")
    return identifier


def _name(value, *, allow_empty=False):
    if not isinstance(value, str):
        raise ValueError("显示名称必须是文字。")
    if any(unicodedata.category(char).startswith('C') for char in value):
        raise ValueError("显示名称不能包含换行或控制字符。")
    value = value.strip()
    if not (0 if allow_empty else 1) <= len(value) <= MAX_DISPLAY_NAME:
        raise ValueError("显示名称须为 1–80 个字符，不能包含换行或控制字符。")
    return value


class _LibraryRoot:
    def __init__(self, root):
        self.root = Path(root).absolute()
        self.resolved = self.root.resolve()

    def check(self):
        info = self.root.lstat()
        if (_linked(info) or not stat.S_ISDIR(info.st_mode)
                or self.root.resolve() != self.resolved):
            raise ValueError("数据目录已改变或包含链接，请重新检查位置。")

    def entry(self, path, *, directory=False, optional=False):
        self.check()
        relative = path.relative_to(self.root)
        if not relative.parts:
            raise ValueError("不能修改数据根目录。")
        current = self.root
        for part in relative.parts:
            current /= part
            try:
                info = current.lstat()
            except FileNotFoundError:
                if optional and current == path:
                    return None
                raise
            if _linked(info):
                raise ValueError("数据文件或目录包含链接，不能执行此操作。")
            if current != path and not stat.S_ISDIR(info.st_mode):
                raise ValueError("数据目录结构已改变。")
        if not path.resolve().is_relative_to(self.resolved):
            raise ValueError("文件不在对应数据目录内。")
        if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
            raise ValueError("数据文件类型不受支持。")
        return info

    def model_files(self, identifier):
        _identifier(identifier)
        paths = [self.root / f'{identifier}.joblib', self.root / f'{identifier}.json']
        for path in paths:
            self.entry(path)
        summary = paths[1]
        if summary.stat().st_size > MAX_MODEL_SUMMARY_BYTES:
            raise ValueError("模型摘要过大，不能通过页面管理。")
        metadata = json.loads(summary.read_text(encoding='utf-8-sig'))
        if not isinstance(metadata, dict):
            raise ValueError("模型摘要格式无效。")
        return paths


def _read_fields(root, path, identifier):
    info = root.entry(path, optional=True)
    default = {'display_name': identifier, 'favorite': False}
    if info is None:
        return default
    if info.st_size > MAX_LIBRARY_BYTES:
        raise ValueError("名称与星标文件过大。")
    payload = json.loads(path.read_text(encoding='utf-8-sig'))
    if (not isinstance(payload, dict) or payload.get('schema_version') != 1
            or payload.get('id') != identifier or type(payload.get('favorite')) is not bool):
        raise ValueError("名称与星标文件格式无效。")
    return {'display_name': _name(payload.get('display_name'), allow_empty=True) or identifier,
            'favorite': payload['favorite']}


class LibraryMetadata:
    def __init__(self, recordings, models):
        self.recordings = _LibraryRoot(recordings)
        self.models = _LibraryRoot(models)

    def _location(self, kind, identifier, *, verify):
        _identifier(identifier)
        if kind in ('recording', 'recordings'):
            root = self.recordings
            folder = root.root / identifier
            root.entry(folder, directory=True)
            if verify:
                root.entry(folder / 'manifest.json')
                if _read_manifest(folder)['id'] != identifier:
                    raise ValueError("录制编号与目录不一致。")
            return root, folder / 'library.json', 'recording'
        if kind in ('model', 'models'):
            root = self.models
            if verify:
                root.model_files(identifier)
            return root, root.root / f'{identifier}.library.json', 'model'
        raise ValueError("请选择录制数据或模型。")

    def _decorate(self, kind, rows):
        result = []
        for row in rows:
            item = dict(row)
            identifier = item['id']
            try:
                root, path, _ = self._location(kind, identifier, verify=False)
                item.update(_read_fields(root, path, identifier))
            except (OSError, ValueError, TypeError) as exc:
                item.update(display_name=identifier, favorite=False,
                            library_warning=f"名称或星标读取失败：{exc}")
            result.append(item)
        return result

    def decorate_recordings(self, rows):
        return self._decorate('recording', rows)

    def decorate_models(self, rows):
        return self._decorate('model', rows)

    def update(self, kind, identifier, *, display_name=_UNSET, favorite=_UNSET):
        if display_name is _UNSET and favorite is _UNSET:
            raise ValueError("请提供要更新的名称或星标状态。")
        if display_name is not _UNSET:
            display_name = _name(display_name, allow_empty=True)
        if favorite is not _UNSET and type(favorite) is not bool:
            raise ValueError("星标状态必须为开启或关闭。")
        root, path, kind = self._location(kind, identifier, verify=True)
        fields = _read_fields(root, path, identifier)
        if display_name is not _UNSET:
            fields['display_name'] = display_name or identifier
        if favorite is not _UNSET:
            fields['favorite'] = favorite
        payload = {'schema_version': 1, 'id': identifier, **fields,
                   'display_name': '' if fields['display_name'] == identifier else fields['display_name']}
        temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
        try:
            root.entry(temporary, optional=True)
            with temporary.open('x', encoding='utf-8', newline='\n') as handle:
                json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
                handle.write('\n')
                handle.flush()
                os.fsync(handle.fileno())
            root.entry(path, optional=True)
            root.entry(temporary)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                root.entry(temporary)
                temporary.unlink()
        return {'ok': True, 'kind': kind, 'id': identifier, **fields}


class ModelDeletion:
    """Delete only a selected model's confirmed artifacts, without loading it."""
    def __init__(self, models):
        self.files = _LibraryRoot(models)
        self._previews = {}

    def _snapshot(self, identifier):
        paths = self.files.model_files(identifier)
        sidecar = self.files.root / f'{identifier}.library.json'
        if self.files.entry(sidecar, optional=True) is not None:
            paths.append(sidecar)
        entries = {path.name: _fingerprint(self.files.entry(path)) for path in paths}
        try:
            display_name = _read_fields(self.files, sidecar, identifier)['display_name']
        except (OSError, ValueError, TypeError):
            # A malformed label must not prevent deleting its explicitly listed
            # file. Its unchanged stat fingerprint is still part of the preview.
            display_name = identifier
        return {'id': identifier, 'display_name': display_name, 'entries': entries}

    def _prune(self):
        now = time.monotonic()
        self._previews = {token: item for token, item in self._previews.items() if item['expires'] > now}

    def preview(self, identifiers):
        identifiers = RecordingDeletion._ids(identifiers)
        if len({identifier.casefold() for identifier in identifiers}) != len(identifiers):
            raise ValueError("删除列表中有重复模型。")
        self._prune()
        snapshots = [self._snapshot(identifier) for identifier in identifiers]
        if len(self._previews) >= 8:
            self._previews.pop(next(iter(self._previews)))
        token = secrets.token_urlsafe(24)
        self._previews[token] = {'expires': time.monotonic() + TTL_SECONDS, 'snapshots': snapshots}
        return {'ok': True, 'token': token, 'expires_in_s': TTL_SECONDS,
                'models': [{'id': item['id'], 'display_name': item['display_name'],
                            'files': sorted(item['entries'])} for item in snapshots]}

    def _preview(self, token):
        self._prune()
        if not isinstance(token, str) or token not in self._previews:
            raise ValueError("删除确认已过期或无效，请重新选择并预览。")
        return self._previews[token]

    def selected(self, token):
        return [item['id'] for item in self._preview(token)['snapshots']]

    def delete(self, token):
        preview = self._preview(token)
        del self._previews[token]
        snapshots = preview['snapshots']
        for item in snapshots:
            if self._snapshot(item['id']) != item:
                raise ValueError(f"模型 {item['id']} 的文件已改变，请重新预览后确认。")
        deleted, errors, unavailable = [], [], []
        for item in snapshots:
            try:
                if self._snapshot(item['id']) != item:
                    raise ValueError("模型文件已改变，请重新预览。")
                # Keep the summary until all other selected artifacts are gone.
                summary = f"{item['id']}.json"
                ordered = sorted(item['entries'], key=lambda name: (name == summary, name))
                for filename in ordered:
                    path = self.files.root / filename
                    if _fingerprint(self.files.entry(path)) != item['entries'][filename]:
                        raise ValueError("模型文件已改变，请重新预览。")
                    path.unlink()
                deleted.append(item['id'])
            except (OSError, ValueError) as exc:
                errors.append(f"{item['id']}：{exc}；可能已删除部分文件，请检查所列模型文件。")
                if not (self.files.root / f"{item['id']}.joblib").is_file():
                    unavailable.append(item['id'])
                break
        return {'ok': not errors, 'deleted': deleted, 'errors': errors,
                'unavailable': unavailable,
                'remaining': [item['id'] for item in snapshots if item['id'] not in deleted]}
