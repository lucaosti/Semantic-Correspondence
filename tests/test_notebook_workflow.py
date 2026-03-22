from __future__ import annotations

from pathlib import Path

from utils.notebook_workflow import (
    NotebookWorkflowConfig,
    PathsConfig,
    build_training_command,
    load_workflow_config,
    save_workflow_config,
)


def test_notebook_workflow_config_roundtrip(tmp_path: Path) -> None:
    config = NotebookWorkflowConfig(
        paths=PathsConfig(repo_root=str(Path(__file__).resolve().parents[1])),
        experiments=(
            {
                "name": "demo",
                "backbone": "dinov2_vitb14",
                "split": "val",
                "use_window_soft_argmax": False,
            },
        ),
    )
    path = tmp_path / "config.yaml"
    save_workflow_config(config, path)
    loaded = load_workflow_config(path)
    assert loaded.runtime.image_height == 784
    assert len(loaded.experiments) == 1
    assert loaded.experiments[0]["name"] == "demo"


def test_build_training_command_contains_script_and_args() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = NotebookWorkflowConfig(paths=PathsConfig(repo_root=str(repo_root)))
    command = build_training_command(config, config.finetune, mode="finetune")
    assert command[0].endswith("train_finetune.py")
    assert "--backbone" in command
    assert "dinov2_vitb14" in command