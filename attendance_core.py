"""Face registration and fast InsightFace attendance recognition."""
from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path
import re
import shutil

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
STUDENTS_DIR = DATA_DIR / "students"
EMBEDDINGS_FILE = DATA_DIR / "face_embeddings.npz"
ATTENDANCE_LOG_FILE = DATA_DIR / "attendance_log.csv"
REPORTS_DIR = DATA_DIR / "monthly_reports"
ATTENDANCE_COLUMNS = ["Date", "Day", "Name", "SAP ID", "Similarity"]


def ensure_storage() -> None:
    STUDENTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def safe_folder_name(sap_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", sap_id.strip())


def safe_file_name(filename: str) -> str:
    """Keep uploaded images inside the student's own storage folder."""
    name = Path(filename).name
    return re.sub(r"[^A-Za-z0-9._-]", "_", name) or "photo.jpg"


def decode_image(raw: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("This file is not a readable image.")
    return image


def get_face_analyzer():
    """Load InsightFace once and use a CPU-friendly detector size."""
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(480, 480))
    return app


def load_database() -> tuple[np.ndarray, list[str], list[str]]:
    if not EMBEDDINGS_FILE.exists():
        return np.empty((0, 512), dtype=np.float32), [], []
    saved = np.load(EMBEDDINGS_FILE, allow_pickle=False)
    return saved["embeddings"], saved["sap_ids"].tolist(), saved["names"].tolist()


def save_database(embeddings: np.ndarray, sap_ids: list[str], names: list[str]) -> None:
    ensure_storage()
    np.savez_compressed(
        EMBEDDINGS_FILE,
        embeddings=embeddings.astype(np.float32),
        sap_ids=np.asarray(sap_ids, dtype=str),
        names=np.asarray(names, dtype=str),
    )


def registered_students() -> pd.DataFrame:
    _, sap_ids, names = load_database()
    return pd.DataFrame({"SAP ID": sap_ids, "Name": names}).drop_duplicates().sort_values(
        "Name", ignore_index=True
    ) if sap_ids else pd.DataFrame(columns=["SAP ID", "Name"])


def register_student(sap_id: str, name: str, uploads, analyzer) -> tuple[int, list[str]]:
    sap_id, name = sap_id.strip(), name.strip()
    if not sap_id or not name:
        raise ValueError("SAP ID and student name are required.")
    if not uploads:
        raise ValueError("Upload at least one clear face photo.")

    vectors: list[np.ndarray] = []
    warnings: list[str] = []
    target = STUDENTS_DIR / safe_folder_name(sap_id)
    target.mkdir(parents=True, exist_ok=True)
    for upload in uploads:
        raw = upload.getvalue()
        image = decode_image(raw)
        faces = analyzer.get(image)
        if len(faces) != 1:
            warnings.append(f"{upload.name}: expected one face, found {len(faces)}; skipped.")
            continue
        vectors.append(faces[0].normed_embedding.astype(np.float32))
        (target / safe_file_name(upload.name)).write_bytes(raw)
    if not vectors:
        raise ValueError("No usable photo was found. Each photo needs exactly one visible face.")

    old_vectors, old_ids, old_names = load_database()
    kept = [i for i, old_id in enumerate(old_ids) if old_id != sap_id]
    all_vectors = ([old_vectors[i] for i in kept] if len(old_vectors) else []) + vectors
    all_ids = [old_ids[i] for i in kept] + [sap_id] * len(vectors)
    all_names = [old_names[i] for i in kept] + [name] * len(vectors)
    save_database(np.asarray(all_vectors, dtype=np.float32), all_ids, all_names)
    return len(vectors), warnings


def recognize_attendance(
    image: np.ndarray,
    analyzer,
    threshold: float = 0.42,
    attendance_date: date | None = None,
):
    """Detect all faces once, match them to the database, and count each SAP ID once."""
    database, sap_ids, names = load_database()
    if not len(database):
        raise ValueError("No students are registered yet.")

    selected_date = attendance_date or date.today()
    annotated = image.copy()
    best_per_student: dict[str, dict] = {}
    unknown_faces = 0

    faces = analyzer.get(image)
    if not faces:
        return pd.DataFrame(columns=ATTENDANCE_COLUMNS), annotated, 0

    face_embeddings = np.asarray(
        [face.normed_embedding.astype(np.float32) for face in faces],
        dtype=np.float32,
    )
    similarity_matrix = database @ face_embeddings.T
    h, w = image.shape[:2]

    for face_index, face in enumerate(faces):
        scores = similarity_matrix[:, face_index]
        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])

        x1, y1, x2, y2 = face.bbox.astype(int).tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if best_score >= threshold:
            sap_id = str(sap_ids[best_index])
            name = names[best_index]
            item = {
                "Date": selected_date.isoformat(),
                "Day": selected_date.strftime("%A"),
                "Name": name,
                "SAP ID": sap_id,
                "Similarity": round(best_score, 3),
            }
            if (
                sap_id not in best_per_student
                or best_score > best_per_student[sap_id]["Similarity"]
            ):
                best_per_student[sap_id] = item
            label, color = f"{name} ({sap_id})", (0, 180, 0)
        else:
            unknown_faces += 1
            label, color = "Unknown", (0, 0, 220)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    rows = sorted(best_per_student.values(), key=lambda row: row["Name"].lower())
    return pd.DataFrame(rows, columns=ATTENDANCE_COLUMNS), annotated, unknown_faces


def excel_bytes(attendance: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        attendance.drop(columns=["Similarity"], errors="ignore").to_excel(writer, index=False, sheet_name="Present Students")
        sheet = writer.sheets["Present Students"]
        for column in sheet.columns:
            sheet.column_dimensions[column[0].column_letter].width = min(max(len(str(cell.value or "")) for cell in column) + 2, 35)
    return output.getvalue()


def load_attendance_log() -> pd.DataFrame:
    if not ATTENDANCE_LOG_FILE.exists():
        return pd.DataFrame(columns=ATTENDANCE_COLUMNS)
    records = pd.read_csv(ATTENDANCE_LOG_FILE, dtype={"SAP ID": str})
    for column in ATTENDANCE_COLUMNS:
        if column not in records:
            records[column] = ""
    records["Date"] = pd.to_datetime(records["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    records["Day"] = pd.to_datetime(records["Date"], errors="coerce").dt.day_name()
    return records[ATTENDANCE_COLUMNS].dropna(subset=["Date", "SAP ID"])


def save_attendance_records(attendance: pd.DataFrame) -> pd.DataFrame:
    """Add one day's results to the permanent log, replacing duplicate student/date rows."""
    ensure_storage()
    updated = pd.concat([load_attendance_log(), attendance[ATTENDANCE_COLUMNS]], ignore_index=True)
    updated = updated.drop_duplicates(subset=["Date", "SAP ID"], keep="last")
    updated = updated.sort_values(["Date", "Name"], ignore_index=True)
    updated.to_csv(ATTENDANCE_LOG_FILE, index=False)
    return updated


def monthly_excel_bytes(year: int, month: int) -> bytes:
    """Build a monthly workbook showing Present/Absent only for recorded dates."""
    records = load_attendance_log()
    period = f"{year:04d}-{month:02d}"
    records = records[records["Date"].str.startswith(period)].copy()

    students = registered_students()
    if students.empty and not records.empty:
        students = records[["SAP ID", "Name"]].drop_duplicates()

    students = (
        students[["Name", "SAP ID"]]
        .drop_duplicates()
        .sort_values("Name", ignore_index=True)
    )

    summary = students.copy()

    # Only create columns for dates on which at least one attendance record exists.
    recorded_dates = sorted(records["Date"].dropna().unique()) if not records.empty else []

    for current_date in recorded_dates:
        present_ids = set(
            records.loc[records["Date"] == current_date, "SAP ID"].astype(str)
        )
        summary[current_date] = (
            summary["SAP ID"]
            .astype(str)
            .isin(present_ids)
            .map({True: "Present", False: "Absent"})
        )

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="Monthly Attendance")
        sheet = writer.sheets["Monthly Attendance"]
        sheet.freeze_panes = "A2"

        for cell in sheet[1]:
            cell.font = cell.font.copy(bold=True)

        for column in sheet.columns:
            width = min(
                max(len(str(cell.value or "")) for cell in column) + 2,
                18,
            )
            sheet.column_dimensions[column[0].column_letter].width = width

    return output.getvalue()


def monthly_report_path(year: int, month: int) -> Path:
    ensure_storage()
    return REPORTS_DIR / f"attendance_{year:04d}_{month:02d}.xlsx"


def save_monthly_workbook(year: int, month: int) -> Path:
    """Refresh the single persistent workbook for this calendar month."""
    report = monthly_report_path(year, month)
    report.write_bytes(monthly_excel_bytes(year, month))
    return report


def clear_database() -> None:
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    ensure_storage()
