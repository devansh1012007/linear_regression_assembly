"""ctypes bindings for libols.so — Phase-1 test entry point."""
import ctypes as C
import numpy as np
from pathlib import Path

LIB = Path(__file__).resolve().parent.parent.parent / "build" / "libols.so"
assert LIB.exists(), f"build the library first: make harness ({LIB})"

ols = C.CDLL(str(LIB))

# double(double*, size_t)
for fn in ("ols_sum_f64_sse2", "ols_sum_f64_avx2"):
    f = getattr(ols, fn)
    f.restype = C.c_double
    f.argtypes = [np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"), C.c_size_t]
# double(double*, double*, size_t)
for fn in ("ols_dot_f64_sse2", "ols_dot_f64_avx2"):
    f = getattr(ols, fn)
    f.restype = C.c_double
    f.argtypes = [np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
                  np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"), C.c_size_t]
# void(double, double*, double*, size_t)
for fn in ("ols_axpy_f64_sse2", "ols_axpy_f64_avx2"):
    f = getattr(ols, fn)
    f.restype = None
    f.argtypes = [C.c_double,
                  np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
                  np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"), C.c_size_t]
# int32(const double*, const double*, size_t, double*, const double*)
ols.ols_centered_sums_f64_sse2.restype = C.c_int32
ols.ols_centered_sums_f64_sse2.argtypes = [
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
    C.c_size_t,
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")]
# int32(x, y, n, flags, out7)
ols.ols_simple_fit_f64_sse2.restype = C.c_int32
ols.ols_simple_fit_f64_sse2.argtypes = [
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
    C.c_size_t, C.c_uint32,
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")]
ols.ols_simple_fit_noint_f64_sse2.restype = C.c_int32
ols.ols_simple_fit_noint_f64_sse2.argtypes = ols.ols_simple_fit_f64_sse2.argtypes
ols.ols_quad_fit_f64_sse2.restype = C.c_int32
ols.ols_quad_fit_f64_sse2.argtypes = ols.ols_simple_fit_f64_sse2.argtypes

# int32(void)
ols.ols_cpu_has_avx2_fma.restype = C.c_int32
ols.ols_cpu_has_avx2_fma.argtypes = []
ols.ols_cpu_has_avx512.restype = C.c_int32
ols.ols_cpu_has_avx512.argtypes = []
ols.ols_isa.restype = C.c_int32
ols.ols_isa.argtypes = []


def centered_sums(x, y, piv=(0.0, 0.0)):
    out = np.zeros(6, dtype=np.float64)
    pivv = np.array(piv, dtype=np.float64)
    rc = ols.ols_centered_sums_f64_sse2(np.ascontiguousarray(x, dtype=np.float64),
                                        np.ascontiguousarray(y, dtype=np.float64),
                                        len(x), out, pivv)
    return rc, out


def simple_fit(x, y, stats=True):
    out = np.zeros(7, dtype=np.float64)
    rc = ols.ols_simple_fit_f64_sse2(np.ascontiguousarray(x, dtype=np.float64),
                                     np.ascontiguousarray(y, dtype=np.float64),
                                     len(x), 1 if stats else 0, out)
    return rc, out


def simple_fit_noint(x, y, stats=True):
    out = np.zeros(5, dtype=np.float64)
    rc = ols.ols_simple_fit_noint_f64_sse2(np.ascontiguousarray(x, dtype=np.float64),
                                           np.ascontiguousarray(y, dtype=np.float64),
                                           len(x), 1 if stats else 0, out)
    return rc, out


def quad_fit(x, y, stats=True):
    out = np.zeros(7, dtype=np.float64)
    rc = ols.ols_quad_fit_f64_sse2(np.ascontiguousarray(x, dtype=np.float64),
                                   np.ascontiguousarray(y, dtype=np.float64),
                                   len(x), 1 if stats else 0, out)
    return rc, out


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000)
    y = 3.0 + 2.5 * x + 0.1 * rng.standard_normal(1000)
    print("isa:", ols.ols_isa(), "(0=sse2 1=avx2 2=avx512)")
    print("sum sse2 :", ols.ols_sum_f64_sse2(x, len(x)), " numpy:", np.sum(x))
    print("sum avx2 :", ols.ols_sum_f64_avx2(x, len(x)))
    print("dot sse2 :", ols.ols_dot_f64_sse2(x, y, len(x)), " numpy:", np.dot(x, y))
    print("dot avx2 :", ols.ols_dot_f64_avx2(x, y, len(x)))
    yy = y.copy()
    ols.ols_axpy_f64_avx2(2.0, x, yy, len(x))
    print("axpy avx2 ok:", np.allclose(yy, y + 2 * x))
    rc, out = simple_fit(x, y)
    print("simple_fit rc:", rc, "b1:", out[0], "b0:", out[1], "r2:", out[4])
    rc, out = simple_fit_noint(x, y)
    print("noint rc:", rc, "b:", out[0])
    rc, out = quad_fit(x, y)
    print("quad rc:", rc, "b:", out[:3])
