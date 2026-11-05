"""Compute only the conditional correct-answer measurements used by the three BPB suites."""

from oe_eval.metrics.metric import MCAccuracy


def gold_request_indices(instances, task_config):
    """Select the same answer position used by native MCAccuracy, keeping every request unchanged."""
    if task_config["primary_metric"] != "bits_per_byte_corr" or not task_config["compute_gold_bpb"]:
        raise ValueError("Correct-answer selection requires the original gold-BPB configuration.")
    if any(instance.request_type != "loglikelihood" for instance in instances):
        raise ValueError("The pinned gold-BPB suites must supply native likelihood requests.")
    positions = list(range(len(instances)))
    offset = (task_config.get("metric_kwargs") or {}).get("uncond_docid_offset")
    if offset:
        middle = len(instances) // 2
        if len(instances) != 2 * middle:
            raise ValueError("Native conditional and unconditional measurements must be paired.")
        for conditioned, unconditioned in zip(instances[:middle], instances[middle:], strict=True):
            if unconditioned.doc_id != conditioned.doc_id + offset or unconditioned.idx != conditioned.idx:
                raise ValueError("The native unconditional measurement order is inconsistent.")
        positions = positions[:middle]
    groups = {}
    for position in positions:
        instance = instances[position]
        groups.setdefault((instance.doc_id, instance.native_id), []).append(position)
    selected = []
    for group in groups.values():
        label = instances[group[0]].label
        if isinstance(label, (str, list)) or label is None:
            label = 0
        try:
            selected.append(group[label])
        except (IndexError, TypeError) as error:
            raise ValueError("The original correct-answer position is missing.") from error
    return selected


class GoldAnswerBPB(MCAccuracy):
    """Reuse native byte conversion and task averaging without computing unused accuracy scores."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uncond_docid_offset = None
        self.metric_names = ["bits_per_byte_corr"]
        self.extra_metric_names = []

    def process_one_doc(self, group_lst):
        if len(group_lst) != 1:
            raise ValueError("Every original example must have exactly one correct-answer measurement.")
        # MCAccuracy.compute_for_requests supplies its original UTF-8 byte conversion constant/formula.
        return {"bits_per_byte_corr": group_lst[0]["model_resps"]["bits_per_byte"]}
