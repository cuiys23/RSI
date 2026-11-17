"""Candidate-side RPC worker. Executed only in an offline Docker container."""
import contextlib
import json
import sys

# Keep the protocol stream separate from candidate print/log output.
protocol_out = sys.stdout
protocol_in = sys.stdin


def request_model(msg, model=None, temperature=0.0, max_tokens=16384, msg_history=None):
    protocol_out.write(json.dumps({'kind': 'llm', 'msg': msg, 'history': msg_history or [],
                                   'temperature': temperature, 'max_tokens': max_tokens}) + '\n')
    protocol_out.flush()
    result = json.loads(protocol_in.readline())
    if 'error' in result:
        raise RuntimeError(result['error'])
    return result['text'], result['history'], result['info']


payload = json.loads(protocol_in.readline())
with contextlib.redirect_stdout(sys.stderr):
    sys.path.insert(0, '/candidate')
    import agent.llm
    agent.llm.get_response_from_llm = request_model
    # Also intercept direct LiteLLM calls; the host still owns model selection.
    def completion(**kwargs):
        messages = kwargs.get('messages', [])
        text, _, info = request_model(
            messages[-1].get('content', ''), msg_history=messages[:-1])
        return {'choices': [{'message': {'content': text}}],
                'model': info.get('actual_model'),
                'usage': {'prompt_tokens': info.get('input_tokens'),
                          'completion_tokens': info.get('output_tokens')}}
    agent.llm.litellm.completion = completion
    from task_agent import TaskAgent
    try:
        agent = TaskAgent(model=payload['model'], chat_history_file='/tmp/history.md')
        prediction, _ = agent.forward(payload['inputs'])
        result = {'kind': 'prediction', 'prediction': prediction}
    except Exception as error:
        result = {'kind': 'prediction', 'prediction': 'None', 'error': str(error)}
protocol_out.write(json.dumps(result) + '\n')
protocol_out.flush()
