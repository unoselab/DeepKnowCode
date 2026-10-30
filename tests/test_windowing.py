"""Tests for token-window coverage and classification input preparation."""
import importlib.util
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / 'src' / 'py' / 'a-06-evaluate-starcoder2.py'
spec = importlib.util.spec_from_file_location('evaluation', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CharacterTokenizer:
    is_fast = True

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def __call__(self, source, add_special_tokens=False, return_offsets_mapping=False):
        result = {'input_ids': self.encode(source, add_special_tokens)}
        if return_offsets_mapping:
            result['offset_mapping'] = [(i, i + 1) for i in range(len(source))]
        return result


def pair(code_a, code_b):
    return {'language': 'java', 'project': 'example', 'row': 0, 'code_a': code_a, 'code_b': code_b}


def coverage(chunks, source):
    mask = [False] * len(source)
    for chunk in chunks:
        a, b = chunk['start'], chunk['end']
        assert chunk['text'] == source[a:b]
        for k in range(a, b):
            mask[k] = True
    return all(mask)


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.tok = CharacterTokenizer()

    def test_short_input_is_unchanged(self):
        item = pair('abc', 'def')
        plan = module.prepare_windows(item, self.tok, 12000, .2)
        self.assertFalse(plan['windowed'])
        prompts = list(module.window_inputs(item, self.tok, plan))
        self.assertEqual(prompts[0][1], self.tok.encode(module.prompt(item)))

    def test_one_long_fragment_is_fully_covered(self):
        item = pair('a' * 18600, 'b' * 250)
        plan = module.prepare_windows(item, self.tok, 12000, .2)
        self.assertTrue(plan['windowed'])
        self.assertTrue(coverage(plan['chunks_a'], item['code_a']))
        self.assertTrue(coverage(plan['chunks_b'], item['code_b']))
        self.assertEqual(plan['window_count'], len(plan['chunks_a']) * len(plan['chunks_b']))
        for _, ids, a, b in module.window_inputs(item, self.tok, plan):
            self.assertLessEqual(len(ids), 12000)
            self.assertEqual(item['code_a'][a['start']:a['end']], a['text'])
            self.assertEqual(item['code_b'][b['start']:b['end']], b['text'])

    def test_two_long_fragments_cartesian_coverage(self):
        item = pair('a' * 9700, 'b' * 9000)
        plan = module.prepare_windows(item, self.tok, 12000, .2)
        self.assertTrue(coverage(plan['chunks_a'], item['code_a']))
        self.assertTrue(coverage(plan['chunks_b'], item['code_b']))
        self.assertGreater(plan['window_count'], 1)
        pairs = list(module.window_inputs(item, self.tok, plan))
        self.assertEqual(len(pairs), plan['window_count'])
        self.assertTrue(all(len(x[1]) <= 12000 for x in pairs))

    def test_window_overlap_no_gaps(self):
        ranges = module.window_ranges(18938, 12000, .2)
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], 18938)
        self.assertTrue(all(ranges[i][1] >= ranges[i+1][0] for i in range(len(ranges)-1)))
        self.assertEqual(len(ranges), 2)

    def test_arbitrary_size_more_than_two_windows(self):
        item = pair('π' * 50000, 'return 1')
        plan = module.prepare_windows(item, self.tok, 12000, .2)
        self.assertTrue(coverage(plan['chunks_a'], item['code_a']))
        self.assertGreaterEqual(plan['window_count'], 5)
        for _, ids, _, _ in module.window_inputs(item, self.tok, plan):
            self.assertLessEqual(len(ids), 12000)



class SummaryTests(unittest.TestCase):
    def test_sensitivity_excludes_windowed_pairs(self):
        import csv
        import json
        import tempfile
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / 'aligned_pairs.jsonl').write_text('{}\n{}\n')
            rows = [
                dict(language='java', project='a', row='0', true_label='1', predicted_label='1', windowed='false', truncated='false', window_count='1'),
                dict(language='java', project='a', row='1', true_label='0', predicted_label='1', windowed='true', truncated='false', window_count='3'),
            ]
            with (out / 'predictions-worker-0.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            args = SimpleNamespace(output_dir=out, num_workers=1, model_id='test', revision='revision', max_input_tokens=12000, window_overlap=.2, aggregation='mean-margin')
            module.summarize(args)
            with (out / 'project_metrics.csv').open(newline='') as f:
                full = list(csv.DictReader(f))[0]
            with (out / 'project_metrics_unwindowed.csv').open(newline='') as f:
                filtered = list(csv.DictReader(f))[0]
            self.assertAlmostEqual(float(full['f1']), 2/3)
            self.assertEqual(float(filtered['f1']), 1)
            self.assertEqual(int(filtered['n_excluded_windowed']), 1)
            self.assertEqual(json.loads((out / 'summary.json').read_text())['total_windows'], 4)

if __name__ == '__main__':
    unittest.main()
