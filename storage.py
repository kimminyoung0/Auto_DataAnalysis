"""프로젝트·분석 이력 디스크 저장."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DATA_ROOT = Path(__file__).parent / "data" / "projects"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _project_dir(project_id: str) -> Path:
    return DATA_ROOT / project_id


def _analysis_dir(project_id: str, analysis_id: str) -> Path:
    return _project_dir(project_id) / "analyses" / analysis_id


def _default_analysis_title(filename: str) -> str:
    stem = Path(filename).stem or "분석"
    local = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"{stem} ({local})"


class ProjectStorage:
    def list_projects(self) -> list[dict[str, Any]]:
        if not DATA_ROOT.exists():
            return []
        projects: list[dict[str, Any]] = []
        for path in sorted(DATA_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not path.is_dir():
                continue
            meta_path = path / "project.json"
            if not meta_path.exists():
                continue
            meta = _read_json(meta_path)
            analyses_dir = path / "analyses"
            count = 0
            if analyses_dir.exists():
                count = sum(1 for a in analyses_dir.iterdir() if a.is_dir())
            meta["analysis_count"] = count
            projects.append(meta)
        return projects

    def create_project(self, name: str) -> dict[str, Any]:
        name = (name or "").strip()
        if not name:
            raise ValueError("프로젝트 이름을 입력하세요.")

        project_id = str(uuid.uuid4())
        now = _utc_now_iso()
        meta = {
            "id": project_id,
            "name": name,
            "created_at": now,
            "updated_at": now,
        }
        _write_json(_project_dir(project_id) / "project.json", meta)
        return meta

    def get_project(self, project_id: str) -> dict[str, Any]:
        meta_path = _project_dir(project_id) / "project.json"
        if not meta_path.exists():
            raise FileNotFoundError("프로젝트를 찾을 수 없습니다.")
        return _read_json(meta_path)

    def touch_project(self, project_id: str) -> None:
        meta_path = _project_dir(project_id) / "project.json"
        if not meta_path.exists():
            return
        meta = _read_json(meta_path)
        meta["updated_at"] = _utc_now_iso()
        _write_json(meta_path, meta)

    def rename_project(self, project_id: str, name: str) -> dict[str, Any]:
        name = (name or "").strip()
        if not name:
            raise ValueError("프로젝트 이름을 입력하세요.")
        meta_path = _project_dir(project_id) / "project.json"
        if not meta_path.exists():
            raise FileNotFoundError("프로젝트를 찾을 수 없습니다.")
        meta = _read_json(meta_path)
        meta["name"] = name
        meta["updated_at"] = _utc_now_iso()
        _write_json(meta_path, meta)
        return meta

    def delete_project(self, project_id: str) -> None:
        path = _project_dir(project_id)
        if not path.exists():
            raise FileNotFoundError("프로젝트를 찾을 수 없습니다.")
        shutil.rmtree(path, ignore_errors=False)

    def list_analyses(self, project_id: str) -> list[dict[str, Any]]:
        self.get_project(project_id)
        analyses_dir = _project_dir(project_id) / "analyses"
        if not analyses_dir.exists():
            return []
        items: list[dict[str, Any]] = []
        for path in analyses_dir.iterdir():
            if not path.is_dir():
                continue
            meta_path = path / "analysis.json"
            if meta_path.exists():
                items.append(_read_json(meta_path))
        items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return items

    def create_analysis(
        self,
        project_id: str,
        *,
        filename: str,
        session_id: str,
        title: str | None = None,
    ) -> dict[str, Any]:
        self.get_project(project_id)
        analysis_id = str(uuid.uuid4())
        now = _utc_now_iso()
        meta = {
            "id": analysis_id,
            "project_id": project_id,
            "title": title or _default_analysis_title(filename),
            "filename": filename,
            "session_id": session_id,
            "created_at": now,
            "updated_at": now,
            "has_report": False,
            "expression": "df",
            "target_col": None,
            "date_col": None,
            "date_start": None,
            "date_end": None,
            "date_filter_active": False,
            "shape": None,
        }
        _write_json(_analysis_dir(project_id, analysis_id) / "analysis.json", meta)
        self.touch_project(project_id)
        return meta

    def get_analysis(self, project_id: str, analysis_id: str) -> dict[str, Any]:
        meta_path = _analysis_dir(project_id, analysis_id) / "analysis.json"
        if not meta_path.exists():
            raise FileNotFoundError("분석 기록을 찾을 수 없습니다.")
        return _read_json(meta_path)

    def update_analysis(
        self, project_id: str, analysis_id: str, **fields: Any
    ) -> dict[str, Any]:
        meta = self.get_analysis(project_id, analysis_id)
        meta.update(fields)
        meta["updated_at"] = _utc_now_iso()
        _write_json(
            _analysis_dir(project_id, analysis_id) / "analysis.json",
            meta,
        )
        self.touch_project(project_id)
        return meta

    def save_dataframe(
        self, project_id: str, analysis_id: str, df: pd.DataFrame
    ) -> None:
        path = _analysis_dir(project_id, analysis_id) / "data.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    def load_dataframe(self, project_id: str, analysis_id: str) -> pd.DataFrame:
        path = _analysis_dir(project_id, analysis_id) / "data.parquet"
        if not path.exists():
            raise FileNotFoundError("저장된 데이터가 없습니다.")
        return pd.read_parquet(path)

    def save_report(
        self, project_id: str, analysis_id: str, report: dict[str, Any]
    ) -> None:
        path = _analysis_dir(project_id, analysis_id) / "report.json"
        _write_json(path, report)

    def load_report(self, project_id: str, analysis_id: str) -> dict[str, Any] | None:
        path = _analysis_dir(project_id, analysis_id) / "report.json"
        if not path.exists():
            return None
        return _read_json(path)

    def delete_analysis(self, project_id: str, analysis_id: str) -> None:
        """분석 폴더 전체 삭제"""
        self.get_project(project_id)
        analysis_path = _analysis_dir(project_id, analysis_id)
        if not analysis_path.exists():
            raise FileNotFoundError("분석 기록을 찾을 수 없습니다.")
        shutil.rmtree(analysis_path, ignore_errors=False)
        self.touch_project(project_id)


store = ProjectStorage()
