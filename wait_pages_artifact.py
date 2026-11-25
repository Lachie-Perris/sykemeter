"""Wait for the uploaded Pages artifact to become visible to the deployment API."""
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def wait_for_artifact(fetch, artifact_id, sleep=time.sleep, attempts=12):
    # Two visible responses avoid proceeding on a single transient API result.
    consecutive = 0
    for attempt in range(attempts):
        artifacts = fetch()
        found = any(str(a['id']) == str(artifact_id)
                    and a['name'] == 'github-pages' and not a.get('expired', False)
                    for a in artifacts)
        consecutive = consecutive + 1 if found else 0
        print(f'Pages artifact check {attempt + 1}/{attempts}: '
              f'{"visible" if found else "not visible yet"}', flush=True)
        if consecutive >= 2:
            return
        if attempt + 1 < attempts:
            sleep(min(2 ** (attempt + 1), 15))
    raise RuntimeError('Uploaded github-pages artifact did not become consistently visible. '
                       'Deployment was not attempted; the published forecast remains unchanged. '
                       'Re-run the workflow to retry.')


def main():
    token = os.environ['GH_TOKEN']
    artifact_id = os.environ['PAGES_ARTIFACT_ID']
    if not artifact_id:
        raise RuntimeError('Pages upload returned no artifact ID.')
    url = (f"{os.environ.get('GITHUB_API_URL', 'https://api.github.com')}/repos/"
           f"{os.environ['GITHUB_REPOSITORY']}/actions/runs/"
           f"{os.environ['GITHUB_RUN_ID']}/artifacts?per_page=100")

    def fetch():
        request = Request(url, headers={'Authorization': f'Bearer {token}',
            'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'})
        try:
            with urlopen(request, timeout=20) as response:
                return json.load(response)['artifacts']
        except HTTPError as exc:
            if exc.code == 429 or exc.code >= 500:
                print(f'GitHub artifact API temporarily returned HTTP {exc.code}', flush=True)
                return []
            raise RuntimeError(f'Artifact API request failed: HTTP {exc.code}') from None
        except (URLError, TimeoutError):
            print('Artifact API temporarily unavailable', flush=True)
            return []

    wait_for_artifact(fetch, artifact_id)


if __name__ == '__main__':
    main()
