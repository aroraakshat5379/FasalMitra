# FasalMitra

Federated multi-crop disease classification pipeline using PlantVillage,
MobileNetV2, and a five-client FedAvg simulation.

## Phase 1: script-based pipeline

### Environment

Use Python 3.10-3.12 and install the pinned dependencies. The TensorFlow
dependency includes CUDA runtime packages for an NVIDIA GPU. For Windows, run
these commands in WSL2 (TensorFlow 2.18 is not supported on native Windows):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv venv "$HOME/.venvs/fasalmitra" --python 3.12
source "$HOME/.venvs/fasalmitra/bin/activate"
cd /path/to/your/repository
uv pip install -r requirements.txt
```

Linux is supported directly.

### Dataset layout

Download and extract PlantVillage locally. Point `--data-dir` to the directory
whose immediate subfolders are class names and whose files are `.jpg`, `.jpeg`,
or `.png` images:

```text
data/PlantVillage/
  Apple___Apple_scab/
    image_001.jpg
  Tomato___Tomato___healthy/
    image_002.jpg
```

The raw dataset is not included. The first run deterministically samples up to
80 images per class, saves a train/test and five-client split manifest, and
reuses that manifest on later runs. This matches the prototype's 80-image
balanced subset and 20% class-stratified global test set. Of each class's
training images, 85% go to its crop-owner client and 15% are distributed among
the other clients, matching the prototype's non-IID overlap. Change the sample
size with `--max-per-class` or pass `--rebuild-data` to regenerate the cached
split. The client crop mapping and all run defaults are in
`fasalmitra/config.py`.

The 38-class Kaggle copy used for this workspace is
[PlantVillage by Mohit Singh](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage)
(Kaggle lists GPL-2). Its root has `train/` and `val/` folders; pass
`--data-dir data/PlantVillage38/PlantVillage`. The pipeline combines those
source folders and creates its own stratified global test split.

### Run

Run commands from the repository root. Each training script can create the
manifest if it does not exist yet:

```powershell
python -m fasalmitra.data --data-dir data/PlantVillage38/PlantVillage
python -m fasalmitra.train_central --data-dir data/PlantVillage38/PlantVillage
python -m fasalmitra.train_fedavg --data-dir data/PlantVillage38/PlantVillage
python -m fasalmitra.evaluate --data-dir data/PlantVillage38/PlantVillage --model results/centralized/centralized_model.keras
```

The default quick-run settings are in `fasalmitra/config.py`. For example:

```powershell
python -m fasalmitra.train_fedavg --data-dir data/PlantVillage38/PlantVillage --rounds 5 --local-epochs 2 --max-per-class 250 --rebuild-data
```

### Phase 2: larger runs and single-image advice

Keep the original 80-image benchmark intact by giving the larger run its own
manifest and results directory. Run centralized training first, then FedAvg
against the same manifest:

```powershell
python -m fasalmitra.train_central --data-dir data/PlantVillage38/PlantVillage --max-per-class 250 --epochs 10 --manifest results/large/data_manifest.json --results-dir results/large/centralized --rebuild-data
python -m fasalmitra.train_fedavg --data-dir data/PlantVillage38/PlantVillage --max-per-class 250 --rounds 15 --local-epochs 2 --manifest results/large/data_manifest.json --results-dir results/large/fedavg
```

Predict a single leaf image with either saved model. Predictions below the
default 65% confidence threshold are withheld and the output asks the farmer to
consult an expert. The bundled seed advisory covers six common tomato and
potato classes; it deliberately does not fabricate advice for the remaining
classes.

```powershell
python -m fasalmitra.predict --image path/to/leaf.jpg --model results/centralized/centralized_model.keras --manifest results/data_manifest.json
```

Use `--threshold 0.75` to require more confidence or `--json` for structured
output. Advisory entries include their TNAU source URL and should be confirmed
with the local agricultural extension service before any treatment.

MobileNetV2 uses ImageNet weights by default and may download them on its first
run. Pass `--weights none` to either training command for offline initialization.

### Outputs

- `results/data_manifest.json`: deterministic sampled train/test and client splits
- `results/class_distribution.png` and `results/client_split.png`: sampled classes and client crop distribution
- `results/centralized/`: model, epoch history, log, summary metrics, class report and confusion matrix
- `results/fedavg/`: model, per-round CSV and log, convergence plot, final class report and confusion matrix
- `results/summary.csv`: centralized/FedAvg accuracy and macro/weighted F1 comparison
- `results/large/summary.csv`: comparison table from the recommended larger run

The data manifest stores paths to the local images; it does not copy or upload
image data. Recreate it if the dataset is moved or changed.

### Current script-pipeline run

On the deterministic 80-images-per-class subset (2,432 train / 608 test), the
centralized baseline reached **0.832 accuracy / 0.828 macro-F1**. Ten-round
FedAvg reached **0.806 / 0.801**. The detailed table and round metrics are in
`results/summary.csv` and `results/fedavg/round_metrics.csv`. These are this
workspace's run results; the figures below are the earlier notebook run.

### Prototype benchmark

The checked-in prototype outputs are preserved under `fasalmitra_results/`.
They report 38 classes and 608 held-out images, with centralized accuracy /
macro-F1 of **0.832 / 0.826** and 10-round FedAvg accuracy / macro-F1 of
**0.791 / 0.785**. These are historical notebook results, not results from
running the new scripts; the script pipeline should be run on the same local
dataset before making a direct comparison.
