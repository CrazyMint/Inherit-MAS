"""Gold-free online WorkBench contract for the native TacoMAS adapter."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from litellm import completion

from tacomas.datasets.base import Dataset, DatasetInstance, DatasetInstanceOutputWithTrajectory
from tacomas.datasets.registry import register_dataset, register_dataset_instance
from tacomas.workbench_bridge import canonical as W


DATASET_ID = "workbench-official"


@register_dataset_instance(DATASET_ID)
class WorkBenchOfficialInstance(DatasetInstance):
    id: str
    question: str
    source: str = ""
    base_template: str = ""

    def get_prompt_info(self) -> Dict[str, str]:
        return {
            "question": f"{W.DATETIME_PREFIX} {self.question}",
            "task_id": self.id,
        }


@register_dataset(DATASET_ID)
class WorkBenchOfficialDataset(Dataset):
    dataset_id: str
    instances: List[WorkBenchOfficialInstance]

    def get_instance_eval_output(
        self,
        instance_output: DatasetInstanceOutputWithTrajectory[WorkBenchOfficialInstance],
    ) -> Dict[str, Any]:
        return {
            "task_id": instance_output.data_instance.id,
            "model_answer": str(instance_output.agent_output or ""),
        }

    def get_instance_eval_metrics(
        self,
        instance_output: DatasetInstanceOutputWithTrajectory[WorkBenchOfficialInstance],
    ) -> Dict[str, Any]:
        instance = instance_output.data_instance
        score, reasoning = self._judge_plan(
            instance.question, str(instance_output.agent_output or "")
        )
        return {
            "success": 1.0 if score >= 0.8 else 0.0,
            "score": score,
            "reasoning": reasoning,
            "online_metric": "gold-free LLM plan judge; official state outcome sealed offline",
        }

    @staticmethod
    def _judge_plan(task: str, candidate: str) -> tuple[float, str]:
        model = os.getenv("JUDGE_MODEL", "openai/gpt-5.4-mini").strip()
        api_key = (
            os.getenv("JUDGE_API_KEY", "").strip()
            or os.getenv("OPENAI_API_KEY", "").strip()
        )
        api_base = (
            os.getenv("JUDGE_API_BASE", "").strip()
            or os.getenv("OPENAI_API_BASE", "").strip()
        )
        if not api_key:
            raise RuntimeError("WorkBench online judge requires JUDGE_API_KEY")
        prompt = f"""You are a strict, gold-free judge of a proposed workplace action plan.
You may use only the public task, the official write-tool signatures, and the
candidate plan below. You do not know the reference actions or expected final
database state. Score whether the plan is executable, uses the right tool and
arguments, resolves every entity/data dependency it claims to resolve, avoids
unrequested writes, and appears to satisfy the task. Do not reward verbosity or
confidence. Return JSON only: {{"score": number from 0 to 1,
"reasoning": "one short diagnostic"}}.

Public task:
{W.DATETIME_PREFIX} {task}

Official write-tool signatures:
{W.write_tool_schema_block()}

Candidate plan:
{candidate}"""
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            api_base=api_base or None,
            temperature=0.0,
            max_tokens=300,
            num_retries=0,
        )
        text = str(response.choices[0].message.content or "")
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise RuntimeError("WorkBench online judge returned no JSON object")
        payload = json.loads(match.group())
        score = max(0.0, min(1.0, float(payload["score"])))
        return score, str(payload.get("reasoning", ""))[:500]

    def get_metrics(self, eval_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not eval_outputs:
            return {"judge_success_rate": 0.0, "num_instances": 0}
        return {
            "judge_success_rate": sum(
                float(row.get("success", 0.0)) for row in eval_outputs
            ) / len(eval_outputs),
            "num_instances": len(eval_outputs),
        }
