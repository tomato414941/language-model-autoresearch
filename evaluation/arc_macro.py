"""Supply the user-approved ARC BPB aggregation missing from pinned OLMES."""

from oe_eval.configs.task_suites import TASK_SUITE_CONFIGS
from oe_eval.tasks import aggregate_tasks

ARC_BPB_SUITE = "arc:rc:bpb::olmes:full"
APPROVED_CONFIG_FIX = {ARC_BPB_SUITE: {"primary_metric": "macro"}}


def configure_aggregation():
    # Rebuild the native derived table after adding only the approved configuration field.
    suite = TASK_SUITE_CONFIGS[ARC_BPB_SUITE]
    suite["primary_metric"] = "macro"
    aggregate_tasks.TASK_GROUPS_RAW[ARC_BPB_SUITE] = {
        "tasks": suite["tasks"], "expected_count": len(suite["tasks"]), "primary_metric": "macro",
    }
    aggregate_tasks.TASK_GROUPS = aggregate_tasks.reorder_dependencies(
        aggregate_tasks.TASK_GROUPS_RAW, "tasks")
