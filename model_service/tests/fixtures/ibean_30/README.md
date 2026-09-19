# tests/fixtures/ibean_30: 30 iBean images (MIT)

This is the CPU fixture for piece 4 (`docs/piece4-work-plan.md`, W1 "Fixture"). The tests run on a CPU and read this folder, never `data/`: spec 001 builds its manifest (US-1.1), and spec 002 extracts features from it with a tiny backbone. It holds 10 images per iBean class and is laid out like `data/raw/`, so the manifest builder reads it the same way it reads the real folder:

    python -m ms.data.manifests build --dataset ibean --raw-root model_service/tests/fixtures/ibean_30 --out <dir>

- `ibean/extracted/<split>/<class>/`: from each class, the source's own `train` .0 to .5, `validation` .0 to .1 and `test` .0 to .1. They are byte-for-byte copies of `data/raw/ibean/extracted/`, which comes from Hugging Face `AI-Lab-Makerere/beans` @ `27aa014`, the lab's own copy of iBean.
- `ibean/SHA256SUMS`: the 30 matching lines of `data/raw/ibean/SHA256SUMS`.
- `ibean/DOWNLOAD.json`: the record's provenance, exactly as `make download DS=ibean` wrote it.
- `ibean_v1.jsonl` and `ibean_v1.meta.json`: the manifest of the 30 images, named as spec 001 names its output. The work plan called this file `manifest.jsonl`. It has 30 rows: healthy and rust split 7/1/2 into train/val/test, and angular leaf spot `holdout_unknown`. The consumers of the fixture read it, for example the feature-cache tests of spec 002.
- `LICENSE-MIT`: the iBean licence (Copyright (c) 2020 AIR Lab Makerere University).

Attribution: Makerere AI Lab (2020). iBean: bean leaf disease images (healthy, angular leaf spot, bean rust) [Data set]. MIT licence. https://github.com/AI-Lab-Makerere/ibean

## How it is made

`../make_ibean_30.py` rebuilds all of it. Run it with `--manifest-only` to rewrite only the manifest from the committed images, which needs no `data/`.

Until the builder exists (W2), the script writes the manifest itself. It writes the rows spec 001 prescribes for this set, and pHash is pinned in FR-006. An independent implementation reproduced every field except `split`, which is the builder's own seeded draw. Once the builder lands, it writes this manifest, and `test_manifests.py` checks that the committed rows match a fresh build in everything but `split`.
