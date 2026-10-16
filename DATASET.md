# Data and evaluator provenance

This study uses only the FALCON H1 human movement-decoding track from DANDI
Dandiset `000954`. The FALCON dataset documentation describes the available files, naming
conventions, and data terms:

<https://snel-repo.github.io/falcon/datasets>

`data_download_manifest.json` lists each source file and its SHA-256 hash. Download and verify those files with
`python download_h1_data.py --output-dir "$FALCON_H1_DATA"`. Set that variable
to an external directory; the reproduction wrapper uses it to build only the
recorded local cache files. The wrapper checks each NWB and cache hash and
fails if the current DANDI asset differs. Do not copy private EvalAI test data
or raw NWB files into this repository.

The package records the evaluator's upstream repository revision, upstream
and vendored source hashes, the upstream MIT license, and any tracked local
patch. This local patch record is required because the data cache records the
loader hash used to create the analyzed arrays.
