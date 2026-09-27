"""CPU ridge-regression baseline, trained by full-batch gradient descent.

This is the portable training path: it runs on any host, which is what
src/train.mojo uses here (this machine has no GPU). src/gpu_kernels.mojo
implements the same matvec/gradient math as real Mojo GPU kernels for use
on accelerator-equipped hosts -- see that file and the README for how the
two relate.
"""


struct RidgeRegression:
    var n_features: Int
    var weights: List[Float32]
    var bias: Float32

    def __init__(out self, n_features: Int):
        self.n_features = n_features
        self.weights = List[Float32](length=n_features, fill=0.0)
        self.bias = 0.0

    def predict_row(self, x: Span[Float32, _], row_offset: Int) -> Float32:
        var acc: Float32 = self.bias
        for j in range(self.n_features):
            acc += x[row_offset + j] * self.weights[j]
        return acc

    def predict(self, x: Span[Float32, _], n_rows: Int) -> List[Float32]:
        var preds = List[Float32](length=n_rows, fill=0.0)
        for i in range(n_rows):
            preds[i] = self.predict_row(x, i * self.n_features)
        return preds

    def fit(
        mut self,
        x: Span[Float32, _],
        y: Span[Float32, _],
        n_rows: Int,
        learning_rate: Float32,
        l2: Float32,
        epochs: Int,
        log_every: Int = 10,
    ):
        var d = self.n_features
        var inv_n = Float32(2.0) / Float32(n_rows)
        for epoch in range(epochs):
            var grad_w = List[Float32](length=d, fill=0.0)
            var grad_b: Float32 = 0.0
            var sq_err: Float32 = 0.0

            for i in range(n_rows):
                var row_offset = i * d
                var pred = self.predict_row(x, row_offset)
                var residual = pred - y[i]
                grad_b += residual
                sq_err += residual * residual
                for j in range(d):
                    grad_w[j] += residual * x[row_offset + j]

            grad_b *= inv_n
            self.bias -= learning_rate * grad_b
            for j in range(d):
                var gw = grad_w[j] * inv_n + Float32(2.0) * l2 * self.weights[j]
                self.weights[j] -= learning_rate * gw

            if log_every > 0 and (epoch % log_every == 0 or epoch == epochs - 1):
                var mse = sq_err / Float32(n_rows)
                print("  epoch", epoch, "mse", mse)
