"""15x15 Renju (exact rules, teachers, environment, data and training)."""
import os

# Many worker processes run small matrix products; one BLAS thread each avoids heavy oversubscription.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
