# Hyperdimensional representations for cross-session intracortical BCI decoding

Reproduction code and machine-readable evidence for the FALCON H1 study. Repository:
[nGQD/Hyperdimensional-representations-for-cross-session-intracortical-brain-computer-interface-decoding](https://github.com/nGQD/Hyperdimensional-representations-for-cross-session-intracortical-brain-computer-interface-decoding).

The archive contains the frozen study code, environment, data and protocol
manifests, compact result evidence, and the selected deployment checkpoint.
It does not redistribute raw recordings. Plotting scripts regenerate figures from the numerical results. The small
`hdc_study/theory.tex` fragment remains because the report step reads it.

## Reproduce

Create the pinned Python 3.10.20 environment. The lockfile uses PyTorch
2.5.1+cu124; GPU runs require a compatible NVIDIA driver.

```sh
conda env create -f environment.lock.yml
conda activate falcon-h1-hdc-study
```

Download the manifest-listed H1 NWB files to an external directory. The
downloader checks every file against its recorded SHA256; follow the dataset's
access terms.

```sh
export FALCON_H1_DATA="$PWD/falcon_h1_data"
python download_h1_data.py --output-dir "$FALCON_H1_DATA"
```

Prepare the verified local caches and run the complete experiment and analysis
workflow:

```sh
python run.py --prepare-cache
python hdc_study/workflow.py
```

The workflow resumes compatible fit records, performs the diagnostics and
post-fit analyses, and regenerates the figures under
`hdc_study/figures/`. It does not need a pre-existing manuscript directory or
PDF bundle; the report step may create local generated text files. A new
checkout starts with an empty runtime fit directory. The archived evidence
under `evidence/` is preserved for inspection and is not used to resume fits.

To generate or verify the Pareto analysis directly from the archived summary
and efficiency records without fitting models:

```sh
python hdc_study/pareto.py --summary evidence/hdc_study/results/summary.json \
  --efficiency evidence/hdc_study/results/efficiency.json
python hdc_study/pareto.py --verify \
  --summary evidence/hdc_study/results/summary.json \
  --efficiency evidence/hdc_study/results/efficiency.json
```

## Evidence and deployment

Archived results and provenance are under `evidence/hdc_study/`. Reproduction
writes new fit records and checkpoints under `hdc_study/results/` and
`hdc_study/checkpoints/`. The only distributed model is
`deployment/transfer_hdc_s101.pt`, paired with its hash-bound JSON record.
The FALCON evaluator source is included under its upstream MIT license; the
study code is AGPL-3.0-only. Dataset access and licensing terms remain those
of the source dataset.

`MANIFEST.sha256` records the archive checksums. Release metadata and the
package status file record author approval and publication status.
