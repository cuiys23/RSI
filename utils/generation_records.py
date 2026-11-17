"""Stable on-disk records for every generation, including failed attempts."""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def evaluation_directory(output_dir, domain, split="train", stage="full", existing=False):
    """Write new IMO runs in the public layout; still read old experiment logs."""
    legacy = f"{domain}_eval" + ("" if split == "train" else f"_{split}")
    if stage == "screen":
        legacy += "_screen"
    if domain != "imo_grading":
        return Path(output_dir) / legacy
    name = "pre_eval" if stage == "screen" else f"{split}_eval"
    canonical = Path(output_dir) / name
    old = Path(output_dir) / legacy
    if existing and not (canonical / "report.json").exists() and old.exists():
        return old
    return canonical


def initialize_record(metadata, domains, meta_model, task_model):
    folder = Path(metadata["gen_output_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    # Do not erase collected artifacts if infrastructure recovery calls us again.
    for name in ("model_patch.diff", "meta_agent_chat.md"):
        (folder / name).touch(exist_ok=True)
    metadata.setdefault("schema_version", 1)
    metadata.setdefault("status", "running")
    metadata.setdefault("started_at", utc_now())
    metadata.setdefault("run_full_eval", False)
    metadata.setdefault("valid_parent", False)
    metadata.setdefault("failure_reason", None)
    metadata.setdefault("models", dict(meta_model=meta_model, task_model=task_model))
    metadata.setdefault("token_usage", {})
    metadata.setdefault("stages", {})
    metadata.setdefault("artifacts", {}).update(
        patch="model_patch.diff", meta_chat="meta_agent_chat.md", log="generate.log")
    if "imo_grading" in domains:
        for split, stage in (("train", "screen"), ("train", "full"), ("val", "full")):
            target = evaluation_directory(folder, "imo_grading", split, stage)
            target.mkdir(exist_ok=True)
            status = target / "stage_status.json"
            if not status.exists():
                write_json(status, dict(stage=stage, split=split, status="not_run",
                                        success=False, reason="Not reached"))
    write_json(folder / "metadata.json", metadata)


def collect_agent_artifacts(metadata):
    """Promote raw container outputs without removing the original evidence."""
    folder = Path(metadata["gen_output_dir"])
    for source, target in (("model_patch.diff", "model_patch.diff"),
                           ("meta_agent_chat_history.md", "meta_agent_chat.md")):
        raw = folder / "agent_output" / source
        if raw.exists():
            shutil.copy2(raw, folder / target)
    patch = folder / "model_patch.diff"
    if patch.exists() and patch.stat().st_size:
        metadata["curr_patch_files"] = [str(patch)]


def finalize_record(metadata, domains):
    folder = Path(metadata["gen_output_dir"])
    collect_agent_artifacts(metadata)
    for split, stage in (("train", "screen"), ("train", "full"), ("val", "full")):
        if "imo_grading" not in domains:
            break
        target = evaluation_directory(folder, "imo_grading", split, stage, existing=True)
        entry = dict(path=str(target.relative_to(folder)), stage=stage, split=split,
                     status="not_run", success=False)
        for name in ("stage_status.json", "report.json"):
            path = target / name
            if path.exists():
                value = json.loads(path.read_text(encoding="utf-8"))
                if name == "stage_status.json":
                    entry.update(value)
                    entry["status"] = value.get("status", "completed" if value.get("success") else "failed")
                else:
                    entry["report"] = value
        metadata.setdefault("stages", {})["pre_eval" if stage == "screen" else f"{split}_eval"] = entry
    if metadata.get("evaluation_error") or metadata.get("parent_agent_success") is False:
        metadata["status"] = "failed"
        metadata["failure_reason"] = metadata.get("evaluation_error") or "Self-modification process failed"
    elif metadata.get("screening_rejected"):
        metadata["status"] = "screening_rejected"
        metadata["failure_reason"] = "No correct prediction in the training screening gate"
    elif metadata.get("run_full_eval"):
        metadata["status"] = "completed"
    elif metadata.get("no_changes"):
        metadata["status"] = "no_changes"
        metadata["failure_reason"] = "Self-modification produced an empty diff"
    else:
        metadata["status"] = "not_evaluated"
    for stage in metadata.get("stages", {}).values():
        if stage.get("status") == "not_run":
            stage["reason"] = metadata.get("failure_reason") or "Stage was not requested"
            write_json(folder / stage["path"] / "stage_status.json", {
                key: value for key, value in stage.items() if key not in {"path", "report"}})
    for category in ("self_modification", "task_evaluation"):
        totals = metadata.setdefault("token_usage", {}).setdefault(category, dict(
            requests=0, input_tokens=0, output_tokens=0, missing_usage=0, failed_requests=0))
        totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
    metadata["completed_at"] = utc_now()
    write_json(folder / "metadata.json", metadata)
    return metadata


def node_status(metadata):
    """Classify both current and historical metadata for the tree legend."""
    if metadata.get("evaluation_error") or metadata.get("parent_agent_success") is False:
        return "failed"
    if metadata.get("screening_rejected"):
        return "screening_rejected"
    if metadata.get("status"):
        return metadata["status"]
    if metadata.get("run_full_eval"):
        return "completed"
    if metadata.get("run_eval") is False:
        return "not_evaluated"
    return "unknown"
