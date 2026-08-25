from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType

KEYPOINT_STRUCT = StructType([
    StructField("x", DoubleType(), True),
    StructField("y", DoubleType(), True),
    StructField("z", DoubleType(), True),
    StructField("score", DoubleType(), True)
])

TELEMETRY_SCHEMA = StructType([
    StructField("session_id", StringType(), False),
    StructField("worker_id", StringType(), False),
    StructField("frame_index", LongType(), False),
    StructField("timestamp_ms", LongType(), False),
    StructField("nose", KEYPOINT_STRUCT, True),
    StructField("l_shoulder", KEYPOINT_STRUCT, True),
    StructField("r_shoulder", KEYPOINT_STRUCT, True),
    StructField("l_hip", KEYPOINT_STRUCT, True),
    StructField("r_hip", KEYPOINT_STRUCT, True),
    StructField("l_elbow", KEYPOINT_STRUCT, True),
    StructField("r_elbow", KEYPOINT_STRUCT, True),
    StructField("l_wrist", KEYPOINT_STRUCT, True),
    StructField("r_wrist", KEYPOINT_STRUCT, True),
    StructField("r_knee", KEYPOINT_STRUCT, True),
    StructField("r_ankle", KEYPOINT_STRUCT, True),
    StructField("ctx_base_silla", LongType(), True),
    StructField("ctx_base_perif", LongType(), True)
])
