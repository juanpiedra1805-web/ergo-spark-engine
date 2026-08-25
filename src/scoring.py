from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# =============================================================================
# 1. MATRIZ OFICIAL MÉTODO ROSA (Tabla E - Sonne et al., 2012)
# =============================================================================
ROSA_TABLA_E = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [2, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [3, 3, 3, 4, 5, 6, 7, 8, 9, 10],
    [4, 4, 4, 4, 5, 6, 7, 8, 9, 10],
    [5, 5, 5, 5, 5, 6, 7, 8, 9, 10],
    [6, 6, 6, 6, 6, 6, 7, 8, 9, 10],
    [7, 7, 7, 7, 7, 7, 7, 8, 9, 10],
    [8, 8, 8, 8, 8, 8, 8, 8, 9, 10],
    [9, 9, 9, 9, 9, 9, 9, 9, 9, 10],
    [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
]

def make_rosa_matrix_udf(spark):
    def rosa_eval(silla_score, perif_score):
        if silla_score is None or perif_score is None:
            return 1
        s_idx = max(0, min(int(silla_score) - 1, 9))
        p_idx = max(0, min(int(perif_score) - 1, 9))
        return int(ROSA_TABLA_E[s_idx][p_idx])
    return F.udf(rosa_eval, IntegerType())

# =============================================================================
# 2. MATRIZ OFICIAL MÉTODO REBA (Tabla C - Hignett & McAtamney, 2000)
# =============================================================================
REBA_TABLA_C = [
    [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
    [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
    [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
    [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
    [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
    [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
    [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
    [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
    [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
    [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
]

def make_reba_matrix_udf(spark):
    def reba_eval(score_a, score_b, actividad=0):
        if score_a is None or score_b is None:
            return 1
        a_idx = max(0, min(int(score_a) - 1, 11))
        b_idx = max(0, min(int(score_b) - 1, 11))
        base_c = REBA_TABLA_C[a_idx][b_idx]
        act_bonus = int(actividad) if actividad is not None else 0
        return int(min(base_c + act_bonus, 15))
    return F.udf(reba_eval, IntegerType())
