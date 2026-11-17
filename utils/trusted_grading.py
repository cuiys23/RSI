"""Trusted host evaluator: labels never enter the candidate container."""
import json
import hashlib
import os
import subprocess
import shutil
import threading
from pathlib import Path

import pandas as pd
from domains.imo.grading_utils import format_input_dict
from domains.report import report
from utils.constants import REPO_NAME
from utils.generation_records import evaluation_directory, utc_now, write_json


def evaluate(container, output_dir, split, stage, model, generation, final=False):
    if split not in {'train', 'val', 'test'} or stage not in {'screen', 'full'}:
        raise ValueError('Invalid evaluation split/stage')
    if split == 'test' and not final:
        raise ValueError('Test data is unavailable during evolution')
    if stage == 'screen' and split != 'train':
        raise ValueError('Only training screening is allowed')
    count = 10 if stage == 'screen' else 100
    folder = evaluation_directory(output_dir, 'imo_grading', split, stage)
    folder.mkdir(parents=True, exist_ok=True)
    status = folder / 'stage_status.json'
    started = utc_now()
    predictions = {}
    errors = []
    data = None
    write_json(status, dict(stage=stage, split=split, status='running', success=False,
                            started_at=started, expected_samples=count))

    def checkpoint():
        if data is not None:
            snapshot = data.copy()
            snapshot['prediction'] = snapshot['Grading ID'].map(predictions)
            snapshot.to_csv(folder / 'predictions.csv', index=False)

    def event(**fields):
        with (folder / 'evaluation.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(timestamp=utc_now(), **fields), ensure_ascii=False) + '\n')

    try:
        data = pd.read_csv(Path(__file__).resolve().parents[1] / 'domains/imo' /
                           f'gradingbench_filtered_100_{split}.csv', dtype=str).iloc[:count].copy()
        if len(data) != count or data['Grading ID'].duplicated().any():
            raise ValueError('Expected fixed, unique evaluation IDs')
        if split == 'train' and stage == 'full':
            screen = evaluation_directory(output_dir, 'imo_grading', 'train', 'screen', existing=True)
            seed = screen / 'predictions.csv'
            if seed.exists():
                previous = pd.read_csv(seed, dtype=str, keep_default_na=False)
                if previous['Grading ID'].tolist() != data['Grading ID'].tolist()[:10]:
                    raise ValueError('Screening IDs do not match fixed training prefix')
                predictions = dict(zip(previous['Grading ID'], previous['prediction']))
                if (screen / 'agent_evals').exists():
                    shutil.copytree(screen / 'agent_evals', folder / 'agent_evals', dirs_exist_ok=True)
                event(event='screening_reused', sample_count=len(predictions))
        # Use the sanitized host mount, never an arbitrary path supplied by Agent.
        mount = next(m for m in container.attrs['Mounts'] if m['Destination'] == f'/{REPO_NAME}')
        candidate = mount['Source']
        image = container.attrs['Config']['Image']
        usage = str(Path(output_dir) / 'task_usage.jsonl')
        for _, row in data.iterrows():
            qid = row['Grading ID']
            if qid in predictions:
                continue
            trace_name = hashlib.sha256(str(qid).encode('utf-8')).hexdigest()[:16] + '.md'
            trace = folder / 'agent_evals' / trace_name
            prediction, error = run_candidate(image, candidate, format_input_dict(row), model,
                                              usage, generation, trace)
            predictions[qid] = prediction
            event(event='prediction', question_id=qid, prediction=prediction, error=error,
                  trace=str(trace.relative_to(folder)))
            if error:
                errors.append({'id': qid, 'error': error})
                if error.startswith('Sandbox failure:'):
                    raise RuntimeError(error)
            checkpoint()
        checkpoint()
        result, _ = report(str(folder), 'imo_grading')
        if result['total'] != count:
            raise ValueError('Evaluation denominator changed')
        write_json(status, dict(stage=stage, split=split, status='completed', success=True,
                               started_at=started, completed_at=utc_now(), expected_samples=count,
                               completed_samples=len(predictions), sample_errors=errors))
        return result
    except (Exception, KeyboardInterrupt) as error:
        checkpoint()
        reason = str(error) or type(error).__name__
        event(event='failure', error=reason)
        write_json(status, dict(stage=stage, split=split, status='failed', success=False,
                               started_at=started, completed_at=utc_now(), expected_samples=count,
                               completed_samples=len(predictions), sample_errors=errors, error=reason))
        raise


def run_candidate(image, candidate, inputs, model, usage, generation, trace_path=None):
    from agent.llm import get_response_from_llm
    worker = str(Path(__file__).with_name('candidate_worker.py').resolve())
    import uuid
    name = 'rsi-eval-' + uuid.uuid4().hex
    def trace(label, value):
        if trace_path is not None:
            path = Path(trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a', encoding='utf-8') as stream:
                stream.write(f'\n## {label}\n\n')
                stream.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2))
                stream.write('\n')

    trace('Task input', inputs)
    command = ['docker', 'run', '--rm', '-i', '--name', name, '--network', 'none',
               '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
               '--pids-limit', '128', '--memory', '4g', '--cpus', '2',
               '--user', '65534:65534', '--tmpfs', '/tmp:rw,nosuid,size=256m',
               '-v', f'{candidate}:/candidate:ro', '-v', f'{worker}:/runner.py:ro',
               '--entrypoint', 'python', image, '-I', '/runner.py']
    import tempfile
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=stderr, text=True, encoding='utf-8')
        timer = threading.Timer(600, lambda: subprocess.run(
            ['docker', 'kill', name], capture_output=True))
        timer.start()
        try:
            process.stdin.write(json.dumps(dict(inputs=inputs, model=model)) + '\n')
            process.stdin.flush()
            calls = 0
            for line in process.stdout:
                message = json.loads(line)
                if message.get('kind') == 'prediction':
                    trace('Prediction', message)
                    prediction = message.get('prediction', 'None')
                    process.stdin.close()
                    code = process.wait(timeout=10)
                    if code:
                        raise RuntimeError(f'Candidate exit code {code}')
                    return prediction, message.get('error')
                if message.get('kind') != 'llm':
                    raise ValueError('Invalid candidate protocol')
                calls += 1
                if calls > 40:
                    raise ValueError('Candidate exceeded request limit')
                # Fixed configuration and accounting belong to the trusted host.
                previous = {key: os.environ.get(key) for key in (
                    'RSI_FIXED_MODEL', 'RSI_USAGE_PATH', 'RSI_GENERATION', 'RSI_CATEGORY')}
                os.environ.update(RSI_FIXED_MODEL=model, RSI_USAGE_PATH=usage,
                                  RSI_GENERATION=str(generation), RSI_CATEGORY='task_evaluation')
                try:
                    trace('Model request', message)
                    text, history, info = get_response_from_llm(
                        message['msg'], model=model, msg_history=message.get('history', []),
                        max_tokens=min(int(message.get('max_tokens', 16384)), 16384))
                    response = dict(text=text, history=history, info=info)
                    trace('Model response', response)
                finally:
                    for key, value in previous.items():
                        if value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = value
                process.stdin.write(json.dumps(response) + '\n')
                process.stdin.flush()
            raise RuntimeError('Candidate exited without a prediction')
        except Exception as error:
            trace('Execution failure', str(error))
            return 'None', 'Sandbox failure: ' + str(error)
        finally:
            timer.cancel()
            subprocess.run(['docker', 'rm', '-f', name], capture_output=True)
            if process.poll() is None:
                process.kill()
            process.wait()
            stderr.seek(0)
            trace('Candidate process log', stderr.read().decode('utf-8', errors='replace'))
