"""
This script shows an example how to use GC-API to programmatically upload
files to an archive.

Remember, an archive is used to pull data per phase for a challenge. Each
phase has a designated archive. The archive is approached using its slug.

For your challenge, and this phase it is 'reg2026-debug-phase-dataset'.

Before you can run this script, you need to:
 * install gc-api (`pip install gcapi`)
 * update the EXPECTED_FILES
 * update the API_TOKEN with a personal token

Get a token from Grand Challenge:
  https://grand-challenge.org/settings/api-tokens/create/

For additional information about using gc-api and tokens, visit:
  https://diagnijmegen.github.io/rse-gcapi/getting-started/

Note that Grand-Challenge does its own post-processing on the files and the archive
will be filled with the result of this post-processing.

Check your uploaded results here:
 https://grand-challenge.org/archives/reg2026-debug-phase-dataset/

And the intermediate processing state here:
  https://grand-challenge.org/cases/uploads/

Happy uploading!
"""

import gcapi

API_TOKEN = "REPLACE-ME-WITH-YOUR-TOKEN"

ARCHIVE_SLUG = "reg2026-debug-phase-dataset"

# Note that SocketValueSpec also allows you to specify
# other sources than `file`, such as multiple `files`, and a direct `value`


EXPECTED_CASE_FILES_FOR_INTERF0 = [
    [
        gcapi.SocketValueSpec(
            socket_slug="visual-context-question", file="visual-context-question.json"
        ),
        gcapi.SocketValueSpec(
            socket_slug="histopathology-region-of-interest-thumbnail",
            file="histopathology-region-of-interest-thumbnail.jpeg",
        ),
    ],
    [
        gcapi.SocketValueSpec(
            socket_slug="visual-context-question", file="visual-context-question.json"
        ),
        gcapi.SocketValueSpec(
            socket_slug="histopathology-region-of-interest-thumbnail",
            file="histopathology-region-of-interest-thumbnail.jpeg",
        ),
    ],
    [
        gcapi.SocketValueSpec(
            socket_slug="visual-context-question", file="visual-context-question.json"
        ),
        gcapi.SocketValueSpec(
            socket_slug="histopathology-region-of-interest-thumbnail",
            file="histopathology-region-of-interest-thumbnail.jpeg",
        ),
    ],
]

EXPECTED_CASE_FILES_FOR_INTERF1 = [
    [
        gcapi.SocketValueSpec(
            socket_slug="whole-slide-image", file="images/whole-slide-image"
        ),
    ],
    [
        gcapi.SocketValueSpec(
            socket_slug="whole-slide-image", file="images/whole-slide-image"
        ),
    ],
    [
        gcapi.SocketValueSpec(
            socket_slug="whole-slide-image", file="images/whole-slide-image"
        ),
    ],
]


EXPECTED_CASES = [
    *EXPECTED_CASE_FILES_FOR_INTERF0,
    *EXPECTED_CASE_FILES_FOR_INTERF1,
]


def main():
    # Uploads files to the Grand-Challenge archive
    total_number_of_cases = len(EXPECTED_CASES)

    with gcapi.Client(token=API_TOKEN) as client:
        for idx, case in enumerate(EXPECTED_CASES):
            print(f"Uploading {idx + 1}/{total_number_of_cases} to {ARCHIVE_SLUG}")
            client.add_case_to_archive(archive_slug=ARCHIVE_SLUG, values=case)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
