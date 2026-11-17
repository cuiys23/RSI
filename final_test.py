"""Explicit, one-shot testing after generation 30. Never resumes search."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import docker
    from utils.constants import REPO_NAME
    from utils.gl_utils import get_patch_files, apply_diffs_container
    from utils.docker_utils import build_container, cleanup_container
    from utils.trusted_grading import evaluate
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    config = json.loads((run / 'experiment_config.json').read_text())
    archive = json.loads((run / 'archive.jsonl').read_text().splitlines()[-1])['archive']
    if not all(i in archive for i in range(31)):
        raise ValueError('Final testing requires baseline 0 and all 30 attempts')
    best = json.loads((run / 'best_agent.json').read_text())
    if best['genid'] is None:
        raise ValueError('No frozen best agent')
    frozen = run / 'final_test'
    frozen.mkdir(exist_ok=True)
    lock = frozen / 'selection.json'
    patches = get_patch_files(str(run), best['genid'])
    selection = dict(best=best, config=config, patches=[
        {'path': str(path), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        for path in patches])
    # Exclusive creation prevents rerunning selection based on observed tests.
    with lock.open('x') as stream:
        json.dump(selection, stream, indent=2)
    results = {}
    root = run / 'gen_initial' / REPO_NAME
    for label, node in [('initial', 0), ('selected', best['genid'])]:
        container = build_container(docker.DockerClient(), str(root), REPO_NAME,
                                    'rsi-final-' + label, domains=['imo_grading'])
        try:
            container.start()
            apply_diffs_container(container, get_patch_files(str(run), node))
            results[label] = evaluate(container, str(frozen / label), 'test', 'full',
                                      config['task_model'], node, final=True)['overall_accuracy']
        finally:
            container.exec_run(['git', 'reset', '--hard', 'HEAD'], workdir=f'/{REPO_NAME}')
            cleanup_container(container)
    results['gain'] = results['selected'] - results['initial']
    (frozen / 'results.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
