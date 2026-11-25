import unittest
from wait_pages_artifact import wait_for_artifact


class ArtifactVisibilityTests(unittest.TestCase):
    artifact = dict(id=123, name='github-pages', expired=False)

    def test_eventual_visibility_after_upload(self):
        results = iter([[], [], [self.artifact], [self.artifact]])
        delays = []
        wait_for_artifact(lambda: next(results), 123, sleep=delays.append)
        self.assertEqual(delays, [2, 4, 8])

    def test_temporary_visibility_does_not_release_deployment(self):
        results = iter([[self.artifact], [], [self.artifact], [self.artifact]])
        wait_for_artifact(lambda: next(results), 123, sleep=lambda _: None)

    def test_wrong_expired_or_missing_artifact_fails(self):
        for artifacts in ([], [dict(self.artifact, id=999)],
                          [dict(self.artifact, name='other')],
                          [dict(self.artifact, expired=True)]):
            with self.subTest(artifacts=artifacts):
                with self.assertRaises(RuntimeError):
                    wait_for_artifact(lambda: artifacts, 123, sleep=lambda _: None, attempts=2)


if __name__ == '__main__':
    unittest.main()
