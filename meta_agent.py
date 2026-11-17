# Copyright (c) Meta Platforms, Inc. and affiliates.

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent

class MetaAgent(AgentSystem):
    def forward(self, repo_path, eval_path, iterations_left=None):
        """
        A meta agent that recursively self-improves.

        Args:
            repo_path (str): The path to the repository.
            eval_path (str): The path to previously generated agents and their evaluation results.
            iterations_left (int, optional): The number of remaining iterations in which the meta agent will be invoked in future. Defaults to None.
        """
        remaining = iterations_left if iterations_left is not None else "unspecified"
        instruction = f"""Task: improve the IMO-GradingBench grading agent.
Objective: maximize exact-match accuracy between predicted scores and expert
scores. Valid output scores are the integers 0, 1, 6, and 7, returned through
{{"response": score}}. Preserve the TaskAgent (prediction, history) interface.

Editable repository: {repo_path}
Historical feedback directory: {eval_path}
Remaining future modification rounds: {remaining}

Inspect historical agent code, patches, evaluation reports (report.json),
predictions (predictions.csv), and error cases in the feedback directory.
Compare predictions with expert Points and use failed question IDs and available
training/validation examples to guide your changes. Evaluate improvements using
the fixed training/validation protocol and overall_accuracy.
Autonomously implement improvements in the repository. Both task_agent.py and
meta_agent.py may be modified, including the strategy used for future improvements.

Constraints:
- Keep the configured models unchanged for both task and meta agents. Do not
  override model arguments or change model selection/configuration.
- Do not modify evaluators, scoring metrics, expert labels, datasets, split
  generation, or the saved split IDs. Do not modify evaluation/search orchestration
  to bypass these constraints or inflate reported results.
- Do not access, inspect, run evaluations on, or use test-set data, labels,
  predictions, or reports. Use only training and validation feedback. If a file
  combines splits, select training/validation IDs without reading test examples.
- Preserve these constraints in any changes to meta_agent.py.
"""

        new_msg_history = chat_with_agent(instruction, model=self.model, msg_history=[], logging=self.log, tools_available='all')
