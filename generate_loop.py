# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import json
import os
import random

from utils.experiment import META_MODEL, TASK_MODEL, MAX_GENERATION, RANDOM_SEED, summarize_usage
from utils.generation_records import (
    collect_agent_artifacts, finalize_record, initialize_record, write_json,
)
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import docker
from analysis.plot_progress import plot_progress_single, plot_progress_together
from analysis.visualize_archive import (
    visualize_archive_single,
    visualize_archive_together,
)

from utils.common import file_exist_and_not_empty, load_json_file
from utils.constants import REPO_NAME
from utils.docker_utils import (
    build_container,
    cleanup_container,
    copy_from_container,
    copy_to_container,
    log_container_output,
    safe_log,
    setup_logger,
)
from utils.domain_utils import (
    can_domain_ensembled,
    get_domain_eval_subset,
    get_domain_splits,
    get_domain_stagedeval_samples,
)
from utils.gl_utils import (
    apply_diffs_container,
    get_patch_files,
    get_score,
    get_full_validation_score,
    load_archive_data,
    run_commands_to_check_compilation,
    select_parent,
    setup_initial_gen,
    update_and_save_archive,
    update_node_metadata,
    get_latest_can_select_parent,
    is_starting_node,
    process_meta_patch_files,
)


def run_harness_polyglot(root_dir, output_dir, genid, skip_staged_eval=False, num_samples=-1):
    # NOTE: the harness for polyglot is different because each task instance needs a docker container
    from domains.polyglot.harness import harness as harness_polyglot
    from domains.polyglot.report import report as report_polyglot

    eval_output_dir = os.path.join(output_dir, f"gen_{genid}", "polyglot_eval")
    test_more_threshold = 0.4  # NOTE: same setting as that in DGM
    model_name_or_path = "eval_run"
    patch_files = get_patch_files(output_dir, genid)
    run_next_eval = True

    # Small sample size evaluation for staged eval
    if not skip_staged_eval:
        test_task_list = load_json_file("./domains/polyglot/subsets/small.json")
        dnames = harness_polyglot(
            test_task_list=test_task_list,
            num_samples=-1,
            max_workers=10,
            model_name_or_path=model_name_or_path,
            model_patch_paths=patch_files,
            num_evals=1,
            num_evals_parallel=1,
            pred_dname=eval_output_dir,
            output_dir=eval_output_dir,
            root_dir=root_dir,
        )
        report_polyglot(output_dir=eval_output_dir, run_keyword=model_name_or_path, expected_num_tasks=len(test_task_list))
        stagedeval_score = get_score("polyglot", output_dir, genid)
        run_next_eval = stagedeval_score is not None and stagedeval_score >= test_more_threshold

    # Check if additional evaluation should be run
    if run_next_eval:
        test_task_list_more = load_json_file("./domains/polyglot/subsets/medium.json")
        dnames = harness_polyglot(
            test_task_list=test_task_list + test_task_list_more,
            num_samples=num_samples,
            max_workers=10,
            model_name_or_path=model_name_or_path,
            model_patch_paths=patch_files,
            num_evals=1,
            num_evals_parallel=1,
            pred_dname=eval_output_dir,
            output_dir=eval_output_dir,
            root_dir=root_dir,
        )
        report_polyglot(output_dir=eval_output_dir, run_keyword=model_name_or_path, expected_num_tasks=len(test_task_list + test_task_list_more))

    # Update metadata
    update_node_metadata(output_dir, genid, {"run_full_eval": run_next_eval})

def select_next_parent_container(
    docker_client,
    domains,
    generate_output_dir,
    archive,
    root_dir="./",
    root_commit="HEAD",
    max_attempts=10,
):
    # Setup logger
    logger = setup_logger(os.path.join(generate_output_dir, "select_next_parent.log"))

    # Get the latest node that can select parent
    latest_node = get_latest_can_select_parent(archive, generate_output_dir)
    safe_log(f"select_next_parent_container: latest_node={latest_node}")
    prev_patch_files = get_patch_files(generate_output_dir, latest_node)

    # Create and start the Docker container
    image_name = f"{REPO_NAME}"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    container_name = f"{REPO_NAME}-nextparent-container-{run_id}"
    container = build_container(docker_client, root_dir, image_name, container_name, verbose=False)
    container.start()
    container_output_folder = "/tmp/"

    try:
        # Apply all lineage diffs
        commit_hash = apply_diffs_container(container, prev_patch_files, verbose=False)

        # Copy generate_output_dir to container
        container_generate_output_dir = os.path.join(
            container_output_folder, generate_output_dir.split(os.sep)[-1]
        )
        copy_to_container(
            container,
            source_path=generate_output_dir,
            dest_path=container_generate_output_dir,
            verbose=False,
        )

        # Get next parent in container
        command = [
            "timeout",
            "3600",  # 1h timeout
            "python",
            "-m",
            "utils.run_select_next_parent",
            "--domains",
            *domains,
            "--generate_output_dir",
            container_generate_output_dir,
        ]
        exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
        log_container_output(exec_result, verbose=True)

        # Get next parent outputs
        container_output_strings = exec_result.output.decode().strip().split("\n")
        next_parent_genid = container_output_strings[-1]
        next_parent_genid = int(next_parent_genid) if not is_starting_node(next_parent_genid) else next_parent_genid

    except Exception as e:
        safe_log(f"Error in select_next_parent_container: {e}")
        update_node_metadata(generate_output_dir, latest_node, {"can_select_next_parent": False})
        next_parent_genid = None

    # Even on errors or KeyboardInterrupt
    finally:
        # Reset to the root commit
        exec_result = container.exec_run(
            cmd=["git", "reset", "--hard", root_commit], workdir=f"/{REPO_NAME}"
        )
        log_container_output(exec_result, verbose=False)
        exec_result = container.exec_run(
            cmd=["git", "clean", "-fd"], workdir=f"/{REPO_NAME}"
        )
        log_container_output(exec_result, verbose=False)

        # Cleanup container
        cleanup_container(container, verbose=False)

    # Try again if no parent is selected
    if next_parent_genid is None:
        if max_attempts > 0:
            next_parent_genid = select_next_parent_container(
                docker_client,
                domains,
                generate_output_dir,
                archive,
                root_dir,
                root_commit,
                max_attempts=max_attempts - 1,
            )
        else:
            # In case of infinite recursive, but it should not happen
            raise Exception("Max attempts reached in select_next_parent_container")

    return next_parent_genid

def get_ensemble_scores_container(
    docker_client,
    domain,
    generate_output_dir,
    gen_output_dir,
    root_dir="./",
    root_commit="HEAD",
    prev_patch_files=[],
    num_samples=-1,
    max_workers=5,
    subsets=[],
):
    # Setup logger
    logger = setup_logger(os.path.join(gen_output_dir, "ensemble.log"))

    # Create and start the Docker container
    image_name = f"{REPO_NAME}"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    container_name = f"{REPO_NAME}-ens-container-{run_id}"
    container = build_container(docker_client, root_dir, image_name, container_name)
    container.start()
    container_output_folder = "/tmp/"

    try:
        # Apply all lineage diffs
        commit_hash = apply_diffs_container(container, prev_patch_files)

        # Copy generate_output_dir to container
        container_generate_output_dir = os.path.join(
            container_output_folder, generate_output_dir.split(os.sep)[-1]
        )
        copy_to_container(
            container,
            source_path=generate_output_dir,
            dest_path=container_generate_output_dir,
        )

        # Get ensemble scores in container
        scores = []
        for subset in subsets:
            command = [
                "timeout",
                "10800",  # 3h timeout
                "python",
                "-m",
                "utils.run_ensemble",
                "--domain",
                domain,
                "--generate_output_dir",
                container_generate_output_dir,
                "--num_samples",
                str(num_samples),
                "--max_workers",
                str(max_workers),
                "--subset",
                subset,
            ]
            exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
            log_container_output(exec_result)

            # Get ensemble outputs
            container_output_strings = exec_result.output.decode().strip().split("\n")
            score = float(container_output_strings[-3])
            scores.append(score)
            container_predictions_path = container_output_strings[-2]
            container_report_path = container_output_strings[-1]

            # Copy container outputs to local
            predictions_file = os.path.basename(container_predictions_path)
            report_file = os.path.basename(container_report_path)
            local_predictions_path = os.path.join(gen_output_dir, predictions_file)
            local_report_path = os.path.join(gen_output_dir, report_file)
            copy_from_container(
                container,
                source_path=container_predictions_path,
                dest_path=local_predictions_path,
            )
            copy_from_container(
                container,
                source_path=container_report_path,
                dest_path=local_report_path,
            )

    except Exception as e:
        safe_log(f"Error in get_ensemble_scores_container: {e}")
        scores = [None] * len(subsets)

    # Even on errors or KeyboardInterrupt
    finally:
        # Reset to the root commit
        exec_result = container.exec_run(
            cmd=["git", "reset", "--hard", root_commit], workdir=f"/{REPO_NAME}"
        )
        log_container_output(exec_result)
        exec_result = container.exec_run(
            cmd=["git", "clean", "-fd"], workdir=f"/{REPO_NAME}"
        )
        log_container_output(exec_result)

        # Cleanup container
        cleanup_container(container)

    return scores


def eval_produced_agent(
    container, container_output_folder, gen_output_dir, domain,
    eval_samples=-1, eval_workers=10, eval_subset="_filtered_100_train",
    split="train", stage="full", task_model=TASK_MODEL, generation=0,
):
    """Evaluate exactly one split; keep screening artifacts separate."""
    if domain == "imo_grading":
        from utils.trusted_grading import evaluate
        return evaluate(container, gen_output_dir, split, stage, task_model, generation)
    if stage not in {"screen", "full"} or split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid evaluation stage/split: {stage}/{split}")
    if stage == "screen" and split != "train":
        raise ValueError("Screening may only evaluate training examples")
    if domain == "imo_grading":
        if split == "test":
            raise ValueError("Test-set evaluation is disabled for IMO grading search")
        eval_samples = 10 if stage == "screen" else 100
    base_run_id = f"{domain}_eval" if split == "train" else f"{domain}_eval_{split}"
    eval_run_id = base_run_id + ("_screen" if stage == "screen" else "")
    container_evaloutput_folder = os.path.join(container_output_folder, eval_run_id)
    local_output = os.path.join(gen_output_dir, eval_run_id)
    os.makedirs(local_output, exist_ok=True)

    def checked(command):
        result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}", environment={
            "RSI_FIXED_MODEL": task_model, "RSI_GENERATION": str(generation),
            "RSI_CATEGORY": "task_evaluation", "RSI_USAGE_PATH": "/tmp/task_usage.jsonl",
        })
        log_container_output(result)
        if result.exit_code != 0:
            raise RuntimeError(f"{domain}/{split}/{stage}: exit {result.exit_code}: {result.output!r}")
        return result

    try:
        # Seed the full training run with all screening artifacts. The harness
        # resumes predictions by ID, including failed attempts marked "None".
        screen_dir = os.path.join(container_output_folder, f"{domain}_eval_screen")
        if stage == "full" and split == "train" and os.path.exists(
            os.path.join(gen_output_dir, f"{domain}_eval_screen", "predictions.csv")
        ):
            checked(["mkdir", "-p", container_evaloutput_folder])
            checked(["cp", "-a", screen_dir + "/.", container_evaloutput_folder])
        subset = eval_subset.replace("_train", f"_{split}")
        checked([
            "timeout", "18000", "python", "-m", "domains.harness",
            "--agent_path", "./task_agent.py", "--output_dir", container_output_folder,
            "--run_id", eval_run_id, "--domain", domain,
            "--num_samples", str(eval_samples), "--num_workers", str(eval_workers),
            "--subset", subset,
        ])
        checked([
            "timeout", "10800", "python", "-m", "domains.report",
            "--domain", domain, "--dname", container_evaloutput_folder,
        ])
        copy_from_container(container, source_path=container_evaloutput_folder,
                            dest_path=local_output)
        report_path = os.path.join(local_output, "report.json")
        with open(report_path) as stream:
            report = json.load(stream)
        # CSV domains must preserve every requested sample and unique ID.
        if domain == "imo_grading" or domain in {"paper_review", "search_arena"}:
            import csv
            from importlib import import_module
            module_name = "domains.imo.grading_utils" if domain == "imo_grading" else f"domains.{domain}.utils"
            id_key = import_module(module_name).QUESTION_ID
            with open(os.path.join(local_output, "predictions.csv"), newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            ids = [row[id_key] for row in rows]
            if len(set(ids)) != len(ids) or any(not qid for qid in ids):
                raise ValueError("Prediction IDs must be unique and nonempty")
            if eval_samples > 0 and len(rows) != eval_samples:
                raise ValueError(f"Expected {eval_samples} predictions, got {len(rows)}")
            if report["total"] != len(rows):
                raise ValueError("Report denominator does not match prediction count")
        with open(os.path.join(local_output, "stage_status.json"), "w") as stream:
            json.dump({"stage": stage, "split": split, "success": True}, stream)
        return report
    except Exception as error:
        with open(os.path.join(local_output, "stage_status.json"), "w") as stream:
            json.dump({"stage": stage, "split": split, "success": False,
                       "error": str(error)}, stream, indent=2)
        raise RuntimeError(f"Evaluation failed ({domain}/{split}/{stage}): {error}") from error



def copy_prev_eval_to_container(
    container,
    prev_eval_path,
    container_output_folder,
    current_genid=None,
    container_folder_name=None,
):
    """Copy the entire prev_eval_path into the container, then remove unwanted files/dirs in the container"""
    if not os.path.exists(prev_eval_path):
        raise FileNotFoundError(f"Previous eval path not found: {prev_eval_path}")

    # Normalize and construct destination path in container
    prev_eval_path = os.path.normpath(prev_eval_path)
    tail = os.path.join(*prev_eval_path.split(os.sep)[-1:])
    container_prev_eval_path = os.path.join(container_output_folder, tail)

    # Ensure destination parent exists
    container.exec_run(["mkdir", "-p", container_output_folder], workdir="/")

    # Copy the whole tree into the container in one go
    import tempfile
    import shutil
    def feedback_ignore(folder, names):
        return [name for name in names if (
            "_test" in name or "_val" in name or name == ".git"
            or name in {"val_eval", "test_eval", "final_test"}
            or name == "grading_split_ids.json" or REPO_NAME == name
            or name.startswith("gradingbench")
        )]
    with tempfile.TemporaryDirectory() as temporary:
        sanitized = os.path.join(temporary, "feedback")
        shutil.copytree(prev_eval_path, sanitized, ignore=feedback_ignore)
        copy_to_container(container, source_path=sanitized, dest_path=container_prev_eval_path)

    # Now prune inside the container
    prune_cmds = [
        # Remove current genid folder
        f"find '{container_prev_eval_path}' -type d -name 'gen_{current_genid}' -prune -exec rm -rf {{}} +",
        # 1) Remove val/test eval directories
        f"find '{container_prev_eval_path}' -type d -name '*_eval_val*' -prune -exec rm -rf {{}} +",
        f"find '{container_prev_eval_path}' -type d -name '*_eval_test*' -prune -exec rm -rf {{}} +",
        # 2) Remove any directories containing the repo name (copied worktrees, etc.)
        f"find '{container_prev_eval_path}' -type d -name '*{REPO_NAME}*' -prune -exec rm -rf {{}} +",
        # 3) Remove compiled Python files
        f"find '{container_prev_eval_path}' -type f -name '*.pyc' -delete",
        # 4) Remove files whose base name indicates val/test (with/without extensions)
        #    *_val, *_val.*, *_val_*, and same for _test
        f"find '{container_prev_eval_path}' -type f \\( -name '*_val' -o -name '*_val.*' -o -name '*_val_*' \\) -delete",
        f"find '{container_prev_eval_path}' -type f \\( -name '*_test' -o -name '*_test.*' -o -name '*_test_*' \\) -delete",
    ]

    for cmd in prune_cmds:
        exec_result = container.exec_run(["bash", "-lc", cmd], workdir="/")

    # Confirm files remaining were copied
    exec_result = container.exec_run(
        ["ls", "-l", container_prev_eval_path], workdir="/"
    )
    log_container_output(exec_result)

    # Move the folder to a new name
    if container_folder_name is not None:
        new_container_prev_eval_path = os.path.join(
            container_output_folder, container_folder_name
        )
        container.exec_run(
            ["mv", container_prev_eval_path, new_container_prev_eval_path], workdir="/"
        )
        log_container_output(exec_result)
        container_prev_eval_path = new_container_prev_eval_path

    return container_prev_eval_path


def generate(
    docker_client,
    domains,
    output_dir,
    run_id,
    current_genid,
    parent_genid,
    root_dir,
    root_commit="main",
    eval_samples=-1,
    eval_workers=10,
    eval_subsets="_filtered_100",
    meta_patch_files=None,
    run_meta_agent=True,
    run_baseline=None,
    optimize_option="only_agent",
    agent_archive_path=None,
    eval_test=False,
    skip_staged_eval=False,
    edit_select_parent=False,
    max_generation=None,
    meta_model=META_MODEL,
    task_model=TASK_MODEL,
):
    # Setup local output folder
    prev_gen_dir = os.path.join(output_dir, f"gen_{parent_genid}")
    gen_output_dir = os.path.join(output_dir, f"gen_{current_genid}")
    os.makedirs(gen_output_dir, exist_ok=True)
    logger = setup_logger(os.path.join(gen_output_dir, "generate.log"))  # Set up logger
    metadata = {
        "gen_output_dir": gen_output_dir,
        "current_genid": current_genid,
        "parent_genid": parent_genid,
        "run_baseline": run_baseline,
        "prev_patch_files": get_patch_files(output_dir, parent_genid),
        "curr_patch_files": [],
        "parent_agent_success": not run_meta_agent,  # meta agent success if not run
        "optimize_option": optimize_option,
        "agent_archive_path": agent_archive_path,
        "can_select_next_parent": True,
    }
    run_eval = not run_meta_agent  # always run eval if not running meta agent
    metadata["run_eval"] = run_eval
    initialize_record(metadata, domains, meta_model, task_model)
    print(metadata)

    # Create and start the Docker container
    image_name = f"{REPO_NAME}"
    container_name = f"{REPO_NAME}-gl-container-{run_id}"
    container = None
    commit_hash = None
    container_agentoutput_folder = "/tmp/agent_output"
    container_output_folder = "/tmp/"

    try:
        container = build_container(docker_client, root_dir, image_name,
                                    container_name, domains=domains)
        if container is None:
            raise RuntimeError("Container creation failed")
        container.start()
        container.exec_run(["touch", "/tmp/meta_usage.jsonl", "/tmp/task_usage.jsonl"])
        # Make a copy of the repo
        if run_baseline and "no_selfimprove" in run_baseline:
            donottouch_reponame = f"/DONOTTOUCH_{REPO_NAME}"
            exec_result = container.exec_run(
                cmd=["cp", "-r", f"/{REPO_NAME}", donottouch_reponame],
                workdir=f"/",
            )
            log_container_output(exec_result)
            meta_patch_files = meta_patch_files or []
            _ = apply_diffs_container(container, meta_patch_files, repo_name=donottouch_reponame)

        # Apply meta patches (only for starting node, because subsequent generations will inherit the patches from the parent)
        if is_starting_node(current_genid):
            meta_patch_files = meta_patch_files or []
            commit_hash = apply_diffs_container(container, meta_patch_files)
            metadata["prev_patch_files"] += meta_patch_files

        # Apply all lineage diffs
        patch_files = get_patch_files(output_dir, parent_genid)
        commit_hash = apply_diffs_container(container, patch_files)

        if run_meta_agent:
            if run_baseline and "dgm" in run_baseline:
                # Get problem statement (DGM specific)
                from baselines.dgm.utils import get_problem_statement
                problem_statement = get_problem_statement(
                    root_dir, output_dir, parent_genid, domains,
                    customized="custom" in run_baseline,
                )

            else:
                # Copy another agent archive to container
                if optimize_option == "only_ensemble":
                    container_agent_archive_path = copy_prev_eval_to_container(
                        container,
                        agent_archive_path,
                        container_output_folder,
                        current_genid=current_genid,
                        container_folder_name="agent_archive",
                    )

                # Copy previous generations to container
                if run_baseline == "no_archive":
                    container_prev_eval_path = os.path.join(
                        container_output_folder, *prev_gen_dir.split(os.sep)[-2:]
                    )
                    copy_to_container(
                        container,
                        source_path=prev_gen_dir,
                        dest_path=container_prev_eval_path,
                    )
                else:
                    container_prev_eval_path = copy_prev_eval_to_container(
                        container, output_dir, container_output_folder, current_genid=current_genid,
                    )

            # Run meta agent
            safe_log("Running meta agent...")
            container_agentoutput_folder = os.path.join(
                container_output_folder, "agent_output"
            )
            container_chat_history_file = os.path.join(
                container_agentoutput_folder, "meta_agent_chat_history.md"
            )
            if run_baseline and "dgm" in run_baseline:
                command = [
                    "timeout",
                    "21600",  # 6h timeout
                    "python",
                    "coding_agent.py",
                    "--problem_statement",
                    problem_statement,
                    "--chat_history_file",
                    container_chat_history_file,
                    "--git_dir",
                    f"/{REPO_NAME}",
                    "--base_commit",
                    commit_hash,
                    "--outdir",
                    container_agentoutput_folder,
                ]
            else:
                command = [
                    "timeout",
                    "21600",  # 6h timeout
                    "python",
                    "run_meta_agent.py",
                    "--chat_history_file",
                    container_chat_history_file,
                    "--repo_path",
                    f"/{REPO_NAME}/",
                    "--evals_folder",
                    container_prev_eval_path,
                    "--git_dir",
                    f"/{REPO_NAME}",
                    "--base_commit",
                    commit_hash,
                    "--outdir",
                    container_agentoutput_folder,
                    "--iterations_left",
                    str(max_generation - current_genid),
                    "--model", meta_model,
                ]

            run_workdir = (
                f"/DONOTTOUCH_{REPO_NAME}"
                if run_baseline and "no_selfimprove" in run_baseline
                else f"/{REPO_NAME}"
            )
            exec_result = container.exec_run(cmd=command, workdir=run_workdir, environment={
                "RSI_FIXED_MODEL": meta_model, "RSI_GENERATION": str(current_genid),
                "RSI_CATEGORY": "self_modification", "RSI_USAGE_PATH": "/tmp/meta_usage.jsonl",
            })
            metadata["parent_agent_success"] = exec_result.exit_code == 0
            log_container_output(exec_result)

            # Restore framework files even when the meta process failed before
            # its own patch-export cleanup. Agent source files remain editable.
            framework_paths = ["agent/llm.py", "utils/experiment.py", "domains/",
                               "run_meta_agent.py", "generate_loop.py", "utils/gl_utils.py",
                               "utils/docker_utils.py", "utils/generation_records.py"]
            result = container.exec_run(
                ["git", "checkout", commit_hash, "--", *framework_paths],
                workdir=f"/{REPO_NAME}",
            )
            log_container_output(result)
            # Copy container outputs to local
            local_agentoutput_folder = os.path.join(gen_output_dir, "agent_output/")
            copy_from_container(
                container,
                source_path=container_agentoutput_folder,
                dest_path=local_agentoutput_folder,
            )

            # Check if agent produced a diff
            local_patch_file = os.path.join(
                local_agentoutput_folder, "model_patch.diff"
            )
            collect_agent_artifacts(metadata)
            run_eval = file_exist_and_not_empty(local_patch_file)
            metadata["run_eval"] = run_eval
            metadata["no_changes"] = not run_eval

            # Run commands to check if the agents are compilable
            if run_eval:
                run_commands_to_check_compilation(container, run_baseline=run_baseline, edit_select_parent=edit_select_parent)

        # Evaluate the produced agent
        if run_eval and "agent" in optimize_option:
            log_path = os.path.join(gen_output_dir, "generate.log")

            def eval_agent_worker(domain, subset, count, split, stage):
                worker_logger = setup_logger(log_path)
                try:
                    return eval_produced_agent(
                        container, container_output_folder, gen_output_dir,
                        domain=domain, eval_samples=count, eval_workers=eval_workers,
                        eval_subset=subset, split=split, stage=stage,
                        task_model=task_model, generation=current_genid,
                    )
                finally:
                    for handler in list(worker_logger.handlers):
                        worker_logger.removeHandler(handler)
                        handler.close()

            metadata["run_full_eval"] = False
            metadata["screening"] = {}
            run_next_eval = True
            if not skip_staged_eval:
                with ThreadPoolExecutor() as executor:
                    futures = {
                        domain: executor.submit(
                            eval_agent_worker, domain, subset,
                            get_domain_stagedeval_samples(domain), "train", "screen",
                        )
                        for domain, subset in zip(domains, eval_subsets)
                    }
                    for domain, future in futures.items():
                        report = future.result()
                        if "total_correct" in report:
                            passed = report["total_correct"] >= 1
                        else:
                            from utils.domain_utils import get_domain_score_key
                            passed = report[get_domain_score_key(domain)] > 0
                        metadata["screening"][domain] = {
                            "passed": passed, "total_correct": report.get("total_correct"),
                        }
                        run_next_eval = run_next_eval and passed
                metadata["screening_rejected"] = not run_next_eval
                if not run_next_eval:
                    metadata["run_eval"] = False
                    safe_log("Screening rejected: at least one domain failed the training gate")

            if run_next_eval:
                counts = eval_samples if isinstance(eval_samples, (list, tuple)) else [eval_samples] * len(domains)
                # Complete training before any validation evaluation begins.
                for split in ("train", "val", "test"):
                    jobs = [
                        (domain, subset, count)
                        for domain, subset, count in zip(domains, eval_subsets, counts)
                        if split in get_domain_splits(domain, eval_test=eval_test)
                        and not (domain == "imo_grading" and split == "test")
                    ]
                    with ThreadPoolExecutor() as executor:
                        futures = [executor.submit(eval_agent_worker, d, s, n, split, "full")
                                   for d, s, n in jobs]
                        for future in futures:
                            future.result()
                metadata["run_full_eval"] = True

    except Exception as e:
        safe_log(f"Error in generate: {e}")
        metadata["run_eval"] = False
        metadata["evaluation_error"] = str(e)
        import traceback
        with open(os.path.join(gen_output_dir, "failure.log"), "a", encoding="utf-8") as stream:
            stream.write(traceback.format_exc())

    except KeyboardInterrupt:
        metadata["run_eval"] = False
        metadata["evaluation_error"] = "Interrupted by user"
        raise

    # Even on errors or KeyboardInterrupt
    finally:
        # Export partial outputs before any reset, including failed meta runs.
        if container is not None and run_meta_agent:
            try:
                copy_from_container(container, source_path=container_agentoutput_folder,
                                    dest_path=os.path.join(gen_output_dir, "agent_output"))
                collect_agent_artifacts(metadata)
            except Exception as error:
                metadata.setdefault("artifact_errors", []).append(str(error))
            if commit_hash and not os.path.exists(os.path.join(gen_output_dir, "agent_output", "model_patch.diff")):
                try:
                    from utils.git_utils import diff_versus_commit
                    patch = diff_versus_commit(root_dir, commit_hash)
                    with open(os.path.join(gen_output_dir, "model_patch.diff"), "w", encoding="utf-8") as stream:
                        stream.write(patch)
                    metadata["curr_patch_files"] = [os.path.join(gen_output_dir, "model_patch.diff")]
                except Exception as error:
                    metadata.setdefault("artifact_errors", []).append(str(error))
        metadata["models"] = {"meta_model": meta_model, "task_model": task_model}
        metadata["token_usage"] = {}
        for category, filename in (("self_modification", "meta_usage.jsonl"),
                                   ("task_evaluation", "task_usage.jsonl")):
            local_usage = os.path.join(gen_output_dir, filename)
            try:
                if container is not None and not (category == "task_evaluation" and "imo_grading" in domains):
                    copy_from_container(container, source_path="/tmp/" + filename, dest_path=local_usage)
                metadata["token_usage"].update(summarize_usage(local_usage))
            except Exception as error:
                metadata.setdefault("accounting_errors", []).append(str(error))
        # Cleanup errors must not erase the record of an attempted generation.
        if container is not None:
            try:
                for command in (["git", "reset", "--hard", root_commit], ["git", "clean", "-fd"]):
                    log_container_output(container.exec_run(cmd=command, workdir=f"/{REPO_NAME}"))
            except Exception as error:
                metadata.setdefault("cleanup_errors", []).append(str(error))
            try:
                cleanup_container(container)
            except Exception as error:
                metadata.setdefault("cleanup_errors", []).append(str(error))

        # Save metadata
        eval_successful = all(
            [
                get_score(domain, output_dir, current_genid) is not None
                for domain in domains
            ]
        )
        # Validation readers inspect metadata too; replace the initial running
        # snapshot before deciding whether this completed candidate is eligible.
        write_json(os.path.join(gen_output_dir, "metadata.json"), metadata)
        if "imo_grading" in domains:
            metadata["valid_parent"] = bool(
                metadata["parent_agent_success"] and metadata["run_eval"]
                and metadata.get("run_full_eval", False)
                and get_full_validation_score("imo_grading", output_dir, current_genid) is not None
            )
        else:
            metadata["valid_parent"] = metadata["run_eval"] and (eval_successful or meta_patch_files is not None)
        try:
            finalize_record(metadata, domains)
        finally:
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.close()

    return metadata


def generate_loop(
    domains,
    run_id=None,
    max_generation=MAX_GENERATION,
    eval_samples=-1,
    eval_workers=5,
    eval_subsets=[],
    parent_selection="score_prop",
    resume_from=None,
    output_dir_parent=None,
    meta_patch_files=None,
    reset_task_agent=False,
    reset_meta_agent=False,
    copy_root_dir=None,  # To have the same initial repo
    run_baseline=None,
    optimize_option="only_agent",
    agent_archive_path=None,
    eval_test=False,
    skip_staged_eval=False,
    edit_select_parent=False,
    meta_model=META_MODEL,
    task_model=TASK_MODEL,
    random_seed=RANDOM_SEED,
):
    # Initialization
    if "imo_grading" in domains and eval_test:
        raise ValueError("eval_test is forbidden during evolution; use frozen final testing")
    if max_generation != MAX_GENERATION:
        raise ValueError("This experiment requires exactly 30 modification rounds")
    random.seed(random_seed)
    import numpy as np
    np.random.seed(random_seed)
    docker_client = docker.DockerClient()
    parent_selection = "latest" if run_baseline == "no_archive" else parent_selection
    if resume_from:
        output_dir = os.path.normpath(os.path.abspath(resume_from))
        if os.path.exists(os.path.join(output_dir, "final_test", "selection.json")):
            raise ValueError("Search is frozen after final test selection")
        with open(os.path.join(output_dir, "experiment_config.json")) as stream:
            saved_config = json.load(stream)
        if saved_config != dict(meta_model=meta_model, task_model=task_model,
                                max_generation=max_generation, random_seed=random_seed):
            raise ValueError("Resume configuration differs from the saved experiment")
        run_id = os.path.basename(output_dir).split("generate_")[-1]
        root_dir, root_commit = setup_initial_gen(
            output_dir,
            domains,
            copy_root_dir=copy_root_dir,
            subsets=eval_subsets,
            resume=True,
            optimize_option=optimize_option,
            run_baseline=run_baseline,
            eval_test=eval_test,
            edit_select_parent=edit_select_parent,
        )
        archive = load_archive_data(
            os.path.join(output_dir, "archive.jsonl"), last_only=True
        )[
            "archive"
        ]  # pyright: ignore
    else:
        run_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S_%f") if run_id is None else run_id
        )
        output_dir_parent = (
            os.path.join(os.getcwd(), "outputs/")
            if output_dir_parent is None
            else output_dir_parent
        )
        output_dir = os.path.normpath(
            os.path.join(output_dir_parent, f"generate_{run_id}/")
        )
        os.makedirs(output_dir, exist_ok=True)
        root_dir, root_commit = setup_initial_gen(
            output_dir,
            domains,
            copy_root_dir=copy_root_dir,
            subsets=eval_subsets,
            resume=False,
            optimize_option=optimize_option,
            run_baseline=run_baseline,
            eval_test=eval_test,
            edit_select_parent=edit_select_parent,
        )

        # Create initial node
        if meta_patch_files is None or len(meta_patch_files) <= 0:
            archive = [] if "imo_grading" in domains else update_and_save_archive(output_dir, [], new_node="initial")
            metadata = {
                "gen_output_dir": os.path.join(output_dir, f"gen_initial"),
                "prev_patch_files": [],
                "curr_patch_files": [],
                "run_eval": True,
            }
        elif reset_task_agent:
            # Task agent is the same as initial agent
            meta_patch_files = process_meta_patch_files(meta_patch_files, output_dir, reset_task_agent=reset_task_agent, reset_meta_agent=reset_meta_agent)
            archive = [] if "imo_grading" in domains else update_and_save_archive(output_dir, [], new_node="initial")
            gen_output_dir = os.path.join(output_dir, f"gen_initial")
            metadata = {
                "gen_output_dir": gen_output_dir,
                "current_genid": "initial",
                "parent_genid": None,
                "run_baseline": run_baseline,
                "prev_patch_files": meta_patch_files,
                "curr_patch_files": [],
                "optimize_option": optimize_option,
                "agent_archive_path": agent_archive_path,
                "can_select_next_parent": True,
                "run_eval": True,
                "run_full_eval": False,
                "valid_parent": True,
            }
            with open(os.path.join(gen_output_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=4)
        else:
            # Process meta patch files
            meta_patch_files = process_meta_patch_files(meta_patch_files, output_dir, reset_task_agent=reset_task_agent, reset_meta_agent=reset_meta_agent)
            # add node 0, which is the evaled version of the patches applied
            archive = update_and_save_archive(output_dir, [], new_node=0)
            metadata = generate(
                docker_client,
                [d for d in domains if d != "polyglot"],
                output_dir,
                run_id,
                current_genid=0,
                parent_genid=None,
                root_dir=root_dir,
                root_commit=root_commit,
                eval_samples=eval_samples,
                eval_workers=eval_workers,
                eval_subsets=eval_subsets,
                meta_patch_files=meta_patch_files,
                run_meta_agent=False,
                run_baseline=run_baseline,
                optimize_option=optimize_option,
                agent_archive_path=agent_archive_path,
                eval_test=eval_test,
                skip_staged_eval=skip_staged_eval,
                edit_select_parent=edit_select_parent,
                max_generation=max_generation,
                meta_model=meta_model, task_model=task_model,
            )
            print(f"generate_loop: generation 0 completed, parent None")
            # Evaluate the agent on polyglot if needed
            if "polyglot" in domains:
                run_harness_polyglot(root_dir, output_dir, 0, skip_staged_eval=skip_staged_eval, num_samples=eval_samples[domains.index("polyglot")])

        # Re-evaluate the initial grading agent as round 0 with the same fixed
        # model and accounting as descendants; imported old scores are not used.
        if "imo_grading" in domains and (not meta_patch_files or reset_task_agent):
            metadata = generate(
                docker_client, ["imo_grading"], output_dir, run_id,
                current_genid=0, parent_genid=None, root_dir=root_dir,
                root_commit=root_commit, eval_samples=[100], eval_workers=eval_workers,
                eval_subsets=["_filtered_100_train"], run_meta_agent=False,
                meta_patch_files=meta_patch_files, skip_staged_eval=True,
                max_generation=max_generation, meta_model=meta_model, task_model=task_model,
            )
            # Round 0 is the baseline, not a modification attempt.
            archive = update_and_save_archive(output_dir, [], new_node=0)

        # Evaluate the entire archive as an ensemble
        eval_ensemble = (
            "ensemble" in optimize_option
            and all(can_domain_ensembled(domain) for domain in domains)
            and run_baseline != "no_archive"
        )
        if metadata["run_eval"] and eval_ensemble:
            for domain, eval_subset, eval_n in zip(domains, eval_subsets, eval_samples):
                _ = get_ensemble_scores_container(
                    docker_client,
                    domain,
                    (
                        output_dir
                        if optimize_option != "only_ensemble"
                        else agent_archive_path
                    ),
                    gen_output_dir=metadata["gen_output_dir"],
                    root_dir=root_dir,
                    root_commit=root_commit,
                    prev_patch_files=metadata["prev_patch_files"]
                    + metadata["curr_patch_files"],
                    num_samples=eval_n,
                    subsets=[
                        eval_subset,
                        eval_subset.replace("_train", "_val"),
                        *(
                            [eval_subset.replace("_train", "_test")]
                            if eval_test
                            else []
                        ),
                    ],
                )

    config = dict(meta_model=meta_model, task_model=task_model,
                  max_generation=max_generation, random_seed=random_seed)
    config_path = os.path.join(output_dir, "experiment_config.json")
    if os.path.exists(config_path):
        with open(config_path) as stream:
            if json.load(stream) != config:
                raise ValueError("Resume configuration differs from the saved experiment")
    else:
        with open(config_path, "w") as stream:
            json.dump(config, stream, indent=2)

    # Save args
    with open(os.path.join(output_dir, "generate_loop.log"), "a") as f:
        args_dict = locals()
        args_str = ", ".join([f"{k}={v}" for k, v in args_dict.items()])
        f.write(f"Args: {args_str}\n")

    # Run generations
    start_genid = max([n for n in archive if isinstance(n, int)] + [0]) + 1
    random.seed(random_seed + start_genid)
    if not edit_select_parent or run_baseline == "no_archive":
        parent_genid = select_parent(archive, output_dir, domains, method=parent_selection)
    else:
        parent_genid = select_next_parent_container(
            docker_client,
            domains,
            output_dir,
            archive,
            root_dir, root_commit,
        )
    for current_genid in range(start_genid, max_generation + 1):
        try:
            metadata = generate(
                docker_client,
                [d for d in domains if d != "polyglot"],
                output_dir,
                run_id,
                current_genid,
                parent_genid=parent_genid,
                root_dir=root_dir,
                root_commit=root_commit,
                eval_samples=eval_samples,
                eval_workers=eval_workers,
                eval_subsets=eval_subsets,
                meta_patch_files=meta_patch_files,
                run_meta_agent=True,
                run_baseline=run_baseline,
                optimize_option=optimize_option,
                agent_archive_path=agent_archive_path,
                eval_test=eval_test,
                skip_staged_eval=skip_staged_eval,
                edit_select_parent=edit_select_parent,
                max_generation=max_generation,
                meta_model=meta_model, task_model=task_model,
            )

        except Exception as error:
            # Infrastructure/cleanup failures still consume an attempted round.
            folder = os.path.join(output_dir, f"gen_{current_genid}")
            os.makedirs(folder, exist_ok=True)
            saved = os.path.join(folder, "metadata.json")
            metadata = load_json_file(saved) if os.path.exists(saved) else {}
            metadata.update(current_genid=current_genid, parent_genid=parent_genid,
                            gen_output_dir=folder, parent_agent_success=False,
                            run_eval=False, valid_parent=False, evaluation_error=str(error))
            metadata.setdefault("prev_patch_files", get_patch_files(output_dir, parent_genid))
            metadata.setdefault("curr_patch_files", [])
            initialize_record(metadata, domains, meta_model, task_model)
            finalize_record(metadata, domains)

        except KeyboardInterrupt:
            # generate() finalized this interrupted attempt before propagating.
            update_and_save_archive(output_dir, archive, new_node=current_genid)
            raise

        # NOTE: need to update and save archive before running ensembling eval
        archive = update_and_save_archive(output_dir, archive, new_node=current_genid)

        # Parent agent failed, update the metadata in the parent node
        if not metadata["parent_agent_success"] and "imo_grading" not in domains:
            update_node_metadata(output_dir, parent_genid, {"valid_parent": False})

        # Evaluate the agent on polyglot if needed
        if "polyglot" in domains:
            run_harness_polyglot(root_dir, output_dir, current_genid, skip_staged_eval=skip_staged_eval, num_samples=eval_samples[domains.index("polyglot")])

        # Evaluate the entire archive as an ensemble
        eval_ensemble = (
            "ensemble" in optimize_option
            and all(can_domain_ensembled(domain) for domain in domains)
            and run_baseline != "no_archive"
        )
        if metadata["run_eval"] and eval_ensemble:
            for domain, eval_subset, eval_n in zip(domains, eval_subsets, eval_samples):
                _ = get_ensemble_scores_container(
                    docker_client,
                    domain,
                    (
                        output_dir
                        if optimize_option != "only_ensemble"
                        else agent_archive_path
                    ),
                    gen_output_dir=metadata["gen_output_dir"],
                    root_dir=root_dir,
                    root_commit=root_commit,
                    prev_patch_files=metadata["prev_patch_files"]
                    + metadata["curr_patch_files"],
                    num_samples=eval_n,
                    subsets=[
                        eval_subset,
                        eval_subset.replace("_train", "_val"),
                        *(
                            [eval_subset.replace("_train", "_test")]
                            if eval_test
                            else []
                        ),
                    ],
                )

        # Make analysis plots
        # Per-domain plots
        for domain in domains:
            splits = get_domain_splits(domain)
            if optimize_option == "only_ensemble":
                score_types = ["ensemble"]
            elif eval_ensemble:
                score_types = ["agent", "ensemble", "max"]
            else:
                score_types = ["agent"]
            for split in splits:  # pyright: ignore
                for stype in score_types:
                    plot_progress_single(domain, output_dir, split=split, type=stype)
                    visualize_archive_single(
                        domain, output_dir, split=split, type=stype
                    )

        # Combined together plots across all domains (if there is more than one domain)
        if len(domains) > 1:
            domain_splits_sets = [set(get_domain_splits(d)) for d in domains]
            common_splits = (
                sorted(list(set.intersection(*domain_splits_sets)))
                if domain_splits_sets
                else []
            )
            if optimize_option == "only_ensemble":
                together_score_types = ["ensemble"]
            elif eval_ensemble:
                together_score_types = ["agent", "ensemble", "max"]
            else:
                together_score_types = ["agent"]
            for split in common_splits:
                for stype in together_score_types:
                    plot_progress_together(domains, output_dir, split=split, type=stype)
                    visualize_archive_together(
                        domains, output_dir, split=split, type=stype
                    )

        # Select next parent; deterministic across stop/resume boundaries.
        random.seed(random_seed + current_genid + 1)
        parent_genid = None
        if not edit_select_parent or run_baseline == "no_archive":
            parent_genid = select_parent(archive, output_dir, domains, method=parent_selection)
        else:
            parent_genid = select_next_parent_container(
                docker_client,
                domains,
                output_dir,
                archive,
                root_dir, root_commit,
            )

        print(f"generate_loop: generation {current_genid} completed, parent {parent_genid}")

    # Best-agent selection uses complete validation accuracy, never training
    # scores or child-count penalties. Every attempted node stays in archive.
    if "imo_grading" in domains:
        scores = {node: get_full_validation_score("imo_grading", output_dir, node)
                  for node in archive}
        eligible = {node: score for node, score in scores.items() if score is not None}
        best = max(eligible, key=eligible.get) if eligible else None
        with open(os.path.join(output_dir, "best_agent.json"), "w") as stream:
            json.dump({"genid": best, "validation_accuracy": eligible.get(best),
                       "metric": "overall_accuracy", "split": "val"}, stream, indent=2)
    # Return output dir
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default=None, help="Run ID")
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",  # one or more domains
        choices=[
            "search_arena",
            "paper_review",
            "balrog_babyai",
            "balrog_babaisai",
            "balrog_minihack",
            "balrog_nle",
            "genesis_go2walking",
            "genesis_go2walkback",
            "genesis_go2hop",
            "polyglot",  # separate harness from the rest
            "imo_grading",
            "imo_proof",
        ],
        required=True,
        help="One or more domains to evaluate (must be from the allowed list)",
    )
    parser.add_argument(
        "--max_generation",
        type=int,
        default=MAX_GENERATION,
        help="Total modification attempts (fixed at 30)",
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        nargs="+",
        default=None,
        help="Evaluation samples per domain (-1 for all). Provide one value per domain.",
    )
    parser.add_argument(
        "--eval_workers",
        type=int,
        default=10,
        help="Number of evaluation workers in parallel",
    )
    parser.add_argument(
        "--parent_selection",
        type=str,
        default="score_child_prop",
        choices=["random", "latest", "best", "score_prop", "score_child_prop"],
        help="Parent selection method",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to an existing output folder to resume from",
    )
    parser.add_argument(
        "--output_dir_parent",
        type=str,
        default=None,
        help="Path to the parent output folder",
    )
    parser.add_argument(
        "--meta_patch_files",
        type=str,
        nargs="+",
        default=[],
        help="Meta patch files to apply",
    )
    parser.add_argument(
        "--reset_task_agent",
        default=False,
        action="store_true",
        help="Whether to reset the changes in the task agent (for self-referential self-improvement transfer experiments)",
    )
    parser.add_argument(
        "--reset_meta_agent",
        default=False,
        action="store_true",
        help="Whether to reset the changes in the meta agent (for self-referential self-improvement transfer experiments)",
    )
    parser.add_argument(
        "--copy_root_dir",
        type=str,
        default=None,
        help="Copy root dir for setup_initial_gen",
    )
    parser.add_argument(
        "--run_baseline",
        type=str,
        choices=[
            "no_selfimprove", "no_archive",
            "dgm", "dgm_custom",
            "dgm+no_selfimprove", "dgm_custom+no_selfimprove",
        ],
        default=None,
        help="Run baseline",
    )
    parser.add_argument(
        "--optimize_option",
        type=str,
        default="only_agent",
        choices=["both_agent_ensemble", "only_agent", "only_ensemble"],
        help="Which part of the algorithm to optimize",
    )
    parser.add_argument(
        "--agent_archive_path",
        type=str,
        default=None,
        help="Path to agent archive (required if --optimize_option=only_ensemble)",
    )
    parser.add_argument(
        "--eval_test",
        default=False,
        action="store_true",
        help="Always run test set evaluation",
    )
    parser.add_argument(
        "--skip_staged_eval",
        default=False,
        action="store_true",
        help="Skip staged evaluation",
    )
    parser.add_argument(
        "--edit_select_parent",
        default=False,
        action="store_true",
        help="Whether to allow the agent to edit the selection mechanism",
    )
    parser.add_argument("--meta_model", default=META_MODEL)
    parser.add_argument("--task_model", default=TASK_MODEL)
    parser.add_argument("--random_seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    # Post-parse validation
    if args.optimize_option == "only_ensemble" and args.agent_archive_path is None:
        parser.error(
            "--agent_archive_path is required when --optimize_option=only_ensemble"
        )
    if args.eval_samples is None:
        eval_samples = [-1] * len(args.domains)
    elif len(args.eval_samples) == len(args.domains):
        eval_samples = args.eval_samples
    else:
        parser.error("--eval_samples must be a one per domain if provided")

    eval_subsets = [get_domain_eval_subset(d) for d in args.domains]
    output_dir = generate_loop(
        domains=args.domains,
        meta_model=args.meta_model, task_model=args.task_model, random_seed=args.random_seed,
        run_id=args.run_id,
        max_generation=args.max_generation,
        eval_samples=eval_samples,
        eval_workers=args.eval_workers,
        eval_subsets=eval_subsets,
        parent_selection=args.parent_selection,
        resume_from=args.resume_from,
        output_dir_parent=args.output_dir_parent,
        meta_patch_files=args.meta_patch_files,
        reset_task_agent=args.reset_task_agent,
        reset_meta_agent=args.reset_meta_agent,
        copy_root_dir=args.copy_root_dir,
        run_baseline=args.run_baseline,
        optimize_option=args.optimize_option,
        agent_archive_path=args.agent_archive_path,
        eval_test=args.eval_test,
        skip_staged_eval=args.skip_staged_eval,
        edit_select_parent=args.edit_select_parent,
    )
