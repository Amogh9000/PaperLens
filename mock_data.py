import numpy as np
import pandas as pd
from datetime import datetime

SUBJECTS = ["Math", "English", "Science", "Reasoning", "GK"]

DUMMY_SINGLE = {
    "student_id": "STU123",
    "subject_scores": {"Math": 18, "English": 15, "Science": 20, "Reasoning": 17, "GK": 16},
    "total_score": 86,
    "annotated_image": None,  # engine will provide a path or bytes later
    "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "confidence_score": 94.5,
}

def make_single(student_id: str = "STU123"):
    scores = {s: int(np.random.randint(10, 21)) for s in SUBJECTS}
    total = int(round(np.clip(np.mean(list(scores.values())) * 5, 0, 100)))
    return {
        "student_id": student_id,
        "subject_scores": scores,
        "total_score": total,
        "annotated_image": None,
        "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "confidence_score": float(np.random.uniform(88, 98)),
    }

def generate_batch(n: int = 12):
    rows = []
    singles = []
    for i in range(n):
        sid = f"STU{i+1:03d}"
        res = make_single(sid)
        singles.append(res)
        row = {"Student_ID": res["student_id"], **{s: res["subject_scores"][s] for s in SUBJECTS}, "Total_Score": res["total_score"]}
        rows.append(row)
    df = pd.DataFrame(rows)
    return df, singles
