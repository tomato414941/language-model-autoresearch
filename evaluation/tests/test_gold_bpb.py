from copy import deepcopy
import unittest

from oe_eval.components.instances import RequestInstance
from oe_eval.components.requests import LoglikelihoodRequest
from oe_eval.metrics.metric import MCAccuracy

from evaluation.gold_bpb import GoldAnswerBPB, gold_request_indices


class CorrectAnswerBPBTests(unittest.TestCase):
    def fixture(self, offset):
        cases = [(2, ["short", "long answer", "日本語🙂"]),
                 (0, ["", "other"]), ("answer", [" answer"]),
                 (["first", "second"], [" first", " second"]), (None, ["code()\n"])]
        instances = []
        for doc_id, (label, choices) in enumerate(cases):
            for index, choice in enumerate(choices):
                instances.append(RequestInstance(
                    request_type="loglikelihood", doc={"choices": choices},
                    request=LoglikelihoodRequest(context="Original question:", continuation=choice),
                    idx=index+7, task_name="synthetic", doc_id=doc_id, native_id=str(doc_id), label=label))
        if offset:
            unconditioned = deepcopy(instances)
            for instance in unconditioned:
                instance.doc_id += offset
                instance.request.context = ""
            instances += unconditioned
        results = []
        for index, instance in enumerate(instances):
            result = instance.to_dict()
            result["model_resps"] = dict(sum_logits=-0.25*(index+1), num_tokens=(index%3)+1)
            results.append(result)
        config = dict(primary_metric="bits_per_byte_corr", compute_gold_bpb=True,
                      metric_kwargs={"uncond_docid_offset":offset})
        return instances, results, config

    def test_preserves_native_per_example_and_task_bpb_for_all_native_label_types(self):
        for offset in (None, 1000000):
            with self.subTest(offset=offset):
                instances, results, config = self.fixture(offset)
                selected = gold_request_indices(instances, config)
                native = MCAccuracy(**config["metric_kwargs"])
                native.compute_for_docs(deepcopy(results))
                expected = native.aggregate_to_task(primary_metric="bits_per_byte_corr")
                optimized = GoldAnswerBPB(**config["metric_kwargs"])
                optimized.compute_for_docs([deepcopy(results[index]) for index in selected])
                actual = optimized.aggregate_to_task(primary_metric="bits_per_byte_corr")
                self.assertEqual(actual["primary_score"], expected["primary_score"])
                self.assertEqual(actual["bits_per_byte_corr"], expected["bits_per_byte_corr"])
                self.assertEqual([row["metrics"]["bits_per_byte_corr"] for row in optimized._scores_for_docs],
                                 [row["metrics"]["bits_per_byte_corr"] for row in native._scores_for_docs])
                self.assertEqual([row["label"] for row in optimized._scores_for_docs],
                                 [row["label"] for row in native._scores_for_docs])

    def test_selects_the_original_choice_position_and_preserves_its_question_and_reference(self):
        instances, _, config = self.fixture(1000000)
        selected = gold_request_indices(instances, config)
        self.assertEqual(selected, [2, 3, 5, 6, 8])
        self.assertEqual(instances[selected[0]].idx, 9)
        self.assertEqual(instances[selected[0]].request.context, "Original question:")
        self.assertEqual(instances[selected[0]].label, 2)
        self.assertEqual(instances[selected[0]].doc["choices"], ["short", "long answer", "日本語🙂"])

    def test_requires_complete_native_pairs_and_a_correct_measurement_for_every_example(self):
        instances, _, config = self.fixture(1000000)
        with self.assertRaises(ValueError):
            gold_request_indices(instances[:-1], config)
        instances[0].label = 100
        with self.assertRaises(ValueError):
            gold_request_indices(instances, config)


if __name__ == "__main__":
    unittest.main()
