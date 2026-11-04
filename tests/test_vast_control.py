import unittest
from unittest.mock import patch

from tools import vast_control


class InstanceListingTests(unittest.TestCase):
    def test_observes_instances_on_every_provider_page(self):
        pages = [
            {'instances': [{'id': 1}], 'total_instances': 2, 'next_token': 'second'},
            {'instances': [{'id': 2}], 'total_instances': 2, 'next_token': None},
        ]
        with patch.object(vast_control, 'api', side_effect=pages) as api:
            self.assertEqual(vast_control.list_instances(), [{'id': 1}, {'id': 2}])
        self.assertEqual(api.call_count, 2)

    def test_reports_an_authoritative_empty_listing(self):
        with patch.object(vast_control, 'api', return_value={
            'instances': [], 'total_instances': 0, 'next_token': None,
        }):
            self.assertEqual(vast_control.list_instances(), [])

    def test_surfaces_failed_incomplete_and_inconsistent_observations(self):
        cases = [
            [{}],
            [{'success': False, 'instances': []}],
            [{'instances': [], 'total_instances': 1, 'next_token': None}],
            [{'instances': [{'id': 1}], 'next_token': 'again'},
             {'instances': [{'id': 1}], 'next_token': None}],
            [{'instances': [], 'next_token': 'again'},
             {'instances': [], 'next_token': 'again'}],
        ]
        for pages in cases:
            with self.subTest(pages=pages), patch.object(vast_control, 'api', side_effect=pages):
                with self.assertRaises(RuntimeError):
                    vast_control.list_instances()


if __name__ == '__main__':
    unittest.main()
