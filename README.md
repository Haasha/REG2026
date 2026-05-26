# Pathologist REasoning-Guided REport Generation Challenge (REG2026)

![REG2026 challenge banner](media/banner.png)

**REG2026** (REG²) is a [Grand Challenge](https://reg2026.grand-challenge.org) competition for pathology **report generation** from gigapixel **whole slide images (WSIs)**, with **diagnostic reasoning** made explicit—not only the final report. It extends REG2025 by evaluating structured reasoning aligned with how pathologists explore slides and decide what is reportable.

The release pairs WSIs (TIFF, 20×) with **chain-of-thought** Q&A and CAP-style report fields (~12K training cases across seven organs). Phases, downloads, rules, and scoring are on the [challenge overview](https://reg2026.grand-challenge.org) and [data description](https://reg2026.grand-challenge.org/data-description/).

## Getting started

This repo holds the **algorithm submission template**, not the dataset. To build and submit a container:

1. Open [`algorithm_submission_template/`](algorithm_submission_template/).
2. Follow [`algorithm_submission_template/GUIDE.md`](algorithm_submission_template/GUIDE.md) (Docker, both task interfaces, local tests, upload).
3. For weights layout, see [`algorithm_submission_template/model/README.md`](algorithm_submission_template/model/README.md).
