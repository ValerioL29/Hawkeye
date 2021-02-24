"""Test cases for mot.py."""
import os
import unittest

from .mot import (
    METRIC_MAPS,
    SUPER_CLASSES,
    acc_single_video,
    aggregate_accs,
    evaluate_mot,
    evaluate_single_class,
    render_results,
)
from .run import read


class TestBDD100KMotEval(unittest.TestCase):
    """Test cases for BDD100K MOT evaluation."""

    def test_mot(self) -> None:
        """Check mot evaluation correctness."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        gts = read("{}/testcases/track_sample_anns/".format(cur_dir))
        preds = read("{}/testcases/track_predictions.json".format(cur_dir))
        result = evaluate_mot(gts, preds)
        overall_reference = {
            "IDF1": 0.7089966679007775,
            "MOTA": 0.6400771952396269,
            "MOTP": 0.8682947680631947,
            "FP": 129,
            "FN": 945,
            "IDSw": 45,
            "MT": 62,
            "PT": 47,
            "ML": 33,
            "FM": 68,
            "mIDF1": 0.3223152925410833,
            "mMOTA": 0.242952917616693,
            "mMOTP": 0.12881014519276474,
        }
        for key in result["OVERALL"]:
            self.assertAlmostEqual(
                result["OVERALL"][key], overall_reference[key]
            )


class TestRenderResults(unittest.TestCase):
    """Test cases for mot render results."""

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    gts = read("{}/testcases/track_sample_anns/".format(cur_dir))
    preds = read("{}/testcases/track_predictions.json".format(cur_dir))

    metrics = list(METRIC_MAPS.keys())
    accs = [acc_single_video(gts[0], preds[0])]
    names, accs, items = aggregate_accs(accs)
    summaries = [
        evaluate_single_class(name, acc) for name, acc in zip(names, accs)
    ]
    eval_results = render_results(summaries, items, metrics)

    def test_categories(self) -> None:
        """Check the correctness of the 1st-level keys in eval_results."""
        cate_names = ["OVERALL"]
        for super_category, categories in SUPER_CLASSES.items():
            cate_names.append(super_category)
            cate_names.extend(categories)

        self.assertEqual(len(self.eval_results), len(cate_names))
        for key in self.eval_results:
            self.assertIn(key, cate_names)

    def test_metrics(self) -> None:
        """Check the correctness of the 2nd-level keys in eval_results."""
        cate_metrics = list(METRIC_MAPS.values())
        overall_metrics = cate_metrics + ["mIDF1", "mMOTA", "mMOTP"]

        for cate, metrics in self.eval_results.items():
            if cate == "OVERALL":
                target_metrics = overall_metrics
            else:
                target_metrics = cate_metrics
            self.assertEqual(len(metrics), len(target_metrics))
            for metric in metrics:
                self.assertIn(metric, target_metrics)


if __name__ == "__main__":
    unittest.main()
