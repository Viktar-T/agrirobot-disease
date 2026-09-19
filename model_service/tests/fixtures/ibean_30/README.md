# tests/fixtures/ibean_30: 30 iBean images (MIT)

This is the CPU fixture for piece 4 (`docs/piece4-work-plan.md`, W1 "Fixture"), used by spec 001 US-1.1. It holds 10 images per iBean class. It is laid out like `data/raw/`, so the manifest builder reads it the same way it reads the real folder:

    python -m ms.data.manifests build --dataset ibean --raw-root model_service/tests/fixtures/ibean_30 --out <dir>

- `ibean/extracted/<split>/<class>/`: from each class, the source's own `train` .0 to .5, `validation` .0 to .1 and `test` .0 to .1. They are byte-for-byte copies of `data/raw/ibean/extracted/`, which comes from Hugging Face `AI-Lab-Makerere/beans` @ `27aa014`, the lab's own copy of iBean.
- `ibean/SHA256SUMS`: the 30 matching lines of `data/raw/ibean/SHA256SUMS`.
- `ibean/DOWNLOAD.json`: the record's provenance, exactly as `make download DS=ibean` wrote it.
- `LICENSE-MIT`: the iBean licence (Copyright (c) 2020 AIR Lab Makerere University).

Attribution: Makerere AI Lab (2020). iBean: bean leaf disease images (healthy, angular leaf spot, bean rust) [Data set]. MIT licence. https://github.com/AI-Lab-Makerere/ibean

`manifest.jsonl`, the golden manifest, comes later: the W2 builder writes it once spec 001 is implemented.
