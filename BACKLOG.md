# Backlog — ML template improvements (Hydra + registry)

Ideas for template extensibility: factory, registry, metrics, config, tracking.
Not tied to a specific domain.

Informed by the research project `papers/schrodingers_negative` (reference implementations in `*/factory.py`).

**Status on `develop` (after merge + registry/metrics/logging):** most MVP items are done. `[x]` = implemented, `[ ]` = still open.

---

## Factory + auto-discovery

- [x] **Shared `registry.py` module**
  `Registry[T]`, `@registry.register("name1", "name2")`, `autodiscover(package, exclude=...)`.
  Factories hold no manual dicts — only `registry.get(config_name)` after lazy discover.

- [x] **Convention: new plugin = module + decorator + YAML**
  Add a dataset / model / metric / tracker without editing central factory code:
  - implementation + `@*_registry.register(...)`
  - file in `conf/<group>/<name>.yaml`
  - switch via CLI: `data=foo`, `model=bar`, `logging=wandb`

- [x] **Registries per layer** (no `eval_metric_registry` yet)

  | Layer | Registry | Example |
  |-------|----------|---------|
  | Dataset | `dataset_registry` | `@dataset_registry.register("mnist")` in `data/mnist.py` |
  | Model | `model_registry` | `@model_registry.register("simple_cnn")` in `models/simple_cnn.py` |
  | Loss | `loss_registry` | `@loss_registry.register("cross_entropy")` in `losses/cross_entropy.py` |
  | Metric (training) | `metric_registry` | `@metric_registry.register("accuracy")` in `metrics/accuracy.py` |
  | Metric (final eval) | `eval_metric_registry` | spec: output fields + deps (e.g. threshold tuning) — **out of scope** |
  | Tracking | `tracking_registry` | `@tracking_registry.register("wandb", "w_and_b")` in `tracking/wandb.py` |

- [x] **Config name normalization**
  `strip`, lowercase, `-` → `_`. Skip values: `none`, `null`, `disabled`.

- [x] **`BuildXxxFn` aliases + `Registry[BuildXxxFn]` per layer**
  Documents the plugin contract and helps mypy/IDE.
  The string in `Registry("…")` is an **error label**, not a config name.

  | Layer | Alias | Plugin signature | Registry |
  |-------|-------|------------------|----------|
  | Dataset | `LoadDataFn` | `(cfg) -> DataBundle` | `dataset_registry` |
  | Model | `BuildModelFn` | `(cfg) -> Model` | `model_registry` |
  | Loss | `BuildLossFn` | `(cfg, num_classes) -> Loss` | `loss_registry` |
  | Metric (training) | `BuildMetricFn` | `(cfg) -> Metric` | `metric_registry` |
  | Metric (eval) | — (spec / dataclass) | output field metadata | `eval_metric_registry` |
  | Tracking | `BuildTrackerFn` | `(cfg) -> TrackingBackend` | `tracking_registry` |

  File convention: `{layer}/{layer}_base.py`, `{layer}/{layer}_factory.py` (e.g. `data/data_base.py`, `metrics/metric_factory.py`).

---

## Metrics

- [x] **Metric phases in config, not one flat list**
  Separate *when* a metric is computed:

  ```yaml
  metrics:
    train:
      active: [loss]
    validation:
      active: [loss, accuracy]
    test:
      active: [accuracy, f1]
  ```

  Empty `active` in a phase → nothing computed for that phase.

- [x] **Config decides *what* to compute, not only *how***
  Final eval respects `metrics.test.active`.
  Metric not in config = not computed, not logged.

- [x] **`MetricsManager` computes only — does not log**
  `compute(phase, context) -> dict[str, float]`.
  `MetricLogger` handles console and trackers.

- [x] **`MetricLogger` — consistent logging moments**

  | Moment | Step | Prefix | Example keys |
  |--------|------|--------|--------------|
  | Run start | `0` | — | `data/*` |
  | Epoch | `epoch` | `train/`, `val/` | `train/loss`, `val/accuracy` |
  | Final eval | `epochs+1` | `test/` | `test/f1` |
  | Summary | none | `summary/` | `summary/best_val_loss` |

- [ ] **Eval metric registry with metadata**
  For metrics outside the training loop: spec (output field names, extra steps e.g. threshold tuning).

- [x] **Validation metrics ≠ final test metrics**
  During epochs: fast val proxies (loss, accuracy).
  After training: full test protocol — different metrics, different prefix, same config-driven selection.

---

## Config (Hydra)

- [x] **One YAML file per plugin**
  `conf/data/mnist.yaml`, `conf/model/cnn.yaml`, `conf/metrics/default.yaml`, `conf/loss/default.yaml` — selected via defaults in `train.yaml`.

- [x] **Eval protocol params in dataset config, not a separate “mode” group**
  Split ratios, seed, batch size — in `conf/data/`.

- [ ] **Per-phase metric config groups**
  Settings under each phase:

  ```yaml
  metrics:
    validation:
      active: [accuracy]
      accuracy:
        top_k: [1, 5]
  ```

  For now, per-metric settings are global (`metrics.f1.average`), not per-phase.

---

## Tracking

- [x] **Composite `TrackingManager`**
  `backends: [wandb, tensorboard]` in config — one API, multiple backends.

- [x] **Tracker auto-discovery**
  `tracking_registry` + decorators in `wandb.py`, `mlflow.py`, `tensorboard.py` — no manual `_BACKEND_MODULES`.

- [x] **Consistent key prefixes**
  `train/`, `val/`, `test/`, `summary/`, `data/` — readable in wandb / tensorboard.

---

## Conscious design choices (out of scope — not planned)

- **PyTorch instead of Lightning** — this template targets structured research with Hydra and plugins (config, registry, tracking). Lightning fits loose notebook prototyping; we keep a thin loop in `trainer.py` so new groups see the full flow without a Trainer abstraction layer.

---

## Future development

Items for later — intentionally omitted from README to keep first clone simple.

- [ ] **Datasets beyond MNIST-family** — examples are grayscale 28×28; other domains need a custom `data/` plugin (machinery exists, no bundled data).
- [ ] **Hydra sweeps** — multirun (`python train.py -m …`) and plugins like Optuna Sweeper can be added via config (`conf/hparams_search/`, `hydra/sweeper`); no presets in the template yet.
- [ ] **Dependent data pipelines** — template assumes a ready dataset and classifier training, not multi-stage ETL.

---

## DevEx

- [x] **ML-oriented `.gitignore`**
  `data/`, `outputs/`, `.venv/`, `wandb/`, `mlruns/`, `.hydra/`, checkpoints (`.pt`, `.pth`, …).

- [x] **Pre-commit + ruff**
  ruff lint/format, yaml — no `conventional-pre-commit` on `commit-msg` yet.

- [x] **README: “Add a plugin in 3 steps”**
  Decorator → YAML → CLI override.

---

## Nice-to-have (low priority)

- [x] Config-driven **optimizer / scheduler**
- [x] **Run info / summary** (`RunSummary` + `summary/*` in tracker)
- [ ] Auto-discovery **eval metric specs** per file
- [x] Smoke test CI: `training.epochs=1` on a small dataset
- [x] Factory tests: mock plugin registration + `build_*` with minimal config

---

## GitHub launch & discoverability

Quick wins before promoting the repo publicly (~15 min). Stars depend on reach, not README length alone.

- [ ] **Enable “Template repository”** in GitHub repo settings — one-click “Use this template” for external groups.
- [ ] **Repository About** — short one-line description under the repo title on GitHub.
- [ ] **Topics** — e.g. `hydra`, `pytorch`, `machine-learning`, `template`, `classification`, `research`, `mlops`.
- [ ] **README screenshot or GIF** — terminal quickstart or a wandb dashboard frame (visual hook for visitors).
- [ ] **Codecov badge** (optional) — upload coverage in CI and add badge to README.
- [ ] **List in awesome-* / course materials** — manual promotion; template is ready, discovery is not automatic.

**Readiness (honest):**

| Goal | ~Status |
|------|---------|
| Research groups / fork | 90% |
| Polished public open source | 80% |
| Viral reach (lightning-hydra scale) | 30% — needs promotion + time |

---

## README one-liner

> Hydra + registry template: plugins register via decorator, config selects *what* to run, `MetricsManager` computes metrics per phase, `MetricLogger` decides *when* and *how* to log to trackers.
