"""Trains, validates, and (if the holdout check passes) exports the Numerai model.

Data loading/splitting (scripts/numerai_data.py) and CORR scoring
(scripts/validate.py) are delegated to Python via Mojo's Python interop, since
they just need pandas/pyarrow and Numerai's own reference scoring code.
The actual training loop -- the part that matters for "utilize the GPU" -- is
native Mojo: src/linear_model.mojo on CPU (used here, since this host has no
accelerator), or src/gpu_kernels.mojo's kernels on a CUDA/ROCm/Metal host.
See README.md for how to switch to the GPU path.
"""
from std.python import Python, PythonObject
from std.python.numpy import from_numpy_array, from_numpy_tensor, copy_to_numpy_array
from std.sys.info import has_accelerator

from linear_model import RidgeRegression

comptime FEATURE_SET = "small"
comptime HOLDOUT_ERAS = 60
comptime EMBARGO_ERAS = 13
comptime LEARNING_RATE: Float32 = 0.05
comptime L2: Float32 = 0.001
comptime EPOCHS = 300
# Sanity bar for "the model learned something real" -- not a claim this beats
# a tuned baseline, just that holdout CORR is positive and not noise-sized.
comptime MIN_MEAN_CORR = 0.0
comptime MIN_SHARPE = 0.3


def main() raises:
    if has_accelerator():
        print("accelerator detected, but this build uses the CPU training path.")
        print("rebuild targeting src/gpu_kernels.mojo to train on the GPU here.")
    else:
        print("no accelerator detected -- training on CPU.")

    Python.add_to_path("scripts")
    var data = Python.import_module("numerai_data")
    var validate = Python.import_module("validate")
    var export = Python.import_module("export_artifacts")

    print("loading data...")
    var split = data.load_split(FEATURE_SET, HOLDOUT_ERAS, EMBARGO_ERAS)
    var feature_set = split[0]
    var fit_x_np = split[1].reshape(-1)
    var fit_y_np = split[2]
    var hold_x_np = split[3]
    var hold_y_np = split[4]
    var hold_eras_np = split[5]
    var mean_np = split[6]
    var std_np = split[7]

    var n_features = len(feature_set)
    var fit_x = from_numpy_array[DType.float32](fit_x_np)
    var fit_y = from_numpy_array[DType.float32](fit_y_np)
    var hold_y = from_numpy_array[DType.float32](hold_y_np)
    var hold_x_view = from_numpy_tensor[DType.float32, 2](hold_x_np)
    var n_fit = len(fit_y)
    var n_hold = len(hold_y)

    print("training ridge regression:", n_fit, "rows x", n_features, "features")
    var model = RidgeRegression(n_features)
    model.fit(
        fit_x, fit_y, n_fit,
        learning_rate=LEARNING_RATE, l2=L2, epochs=EPOCHS, log_every=50,
    )

    print("scoring holdout:", n_hold, "rows")
    var hold_preds = List[Float32](length=n_hold, fill=0.0)
    for i in range(n_hold):
        var acc: Float32 = model.bias
        for j in range(n_features):
            acc += hold_x_view[i, j] * model.weights[j]
        hold_preds[i] = acc

    var hold_preds_np = copy_to_numpy_array(Span(hold_preds))
    var metrics = validate.per_era_corr(
        hold_preds_np, hold_y_np, hold_eras_np,
        min_mean=Float64(MIN_MEAN_CORR), min_sharpe=Float64(MIN_SHARPE),
    )
    print("holdout CORR mean=", metrics["mean"], "std=", metrics["std"], "sharpe=", metrics["sharpe"])

    if Bool(metrics["passed"]):
        print("holdout check passed -- exporting artifacts for submission.")
        var weights_np = copy_to_numpy_array(Span(model.weights))
        export.save(weights_np, Float64(model.bias), feature_set, metrics, mean_np, std_np)
    else:
        raise Error(
            "holdout check failed (mean CORR <= "
            + String(MIN_MEAN_CORR)
            + " or sharpe <= "
            + String(MIN_SHARPE)
            + ") -- refusing to export. Not submitting."
        )
