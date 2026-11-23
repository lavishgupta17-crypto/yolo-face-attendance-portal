# YOLO Attendance Portal

A local Streamlit app that uses **YOLOv8** to locate people in a classroom photo and **InsightFace** to recognize registered students. It keeps an ongoing monthly Excel attendance sheet with date, day, name, and SAP ID.

## Before you start

- Get consent from students before collecting biometric face data.
- Use 5–15 sharp, well-lit photos for every student, with a single face per photo.
- This is an attendance aid: review low-confidence or unknown results before finalizing attendance.

## Run it (Windows)

Open PowerShell in this folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The first use downloads the YOLO and InsightFace model weights, so it needs an internet connection once. Later runs use the locally cached model files.

## Workflow

1. Open **Register students** and enter a name, SAP ID, and reference photos.
2. Open **Take attendance**, upload a single classroom photo, and select a match threshold.
3. Select the attendance date, inspect the annotated image and table, then download the monthly Excel sheet.

The portal creates one persistent monthly workbook in `data/monthly_reports/` (for example, `attendance_2026_08.xlsx`). Every time a classroom photo is processed, it updates that same workbook for the selected month. It has one **Monthly Attendance** sheet: Name, SAP ID, then columns from the first through the last calendar date of the month with weekdays in the headers. `Present` is written for each attended day. Re-processing a student's photo for the same date updates that student's record instead of adding a duplicate.

Student photos and face embeddings remain in `data/` on this computer. The **Database** tab can delete them all.

## Sample dataset (optional, for testing)

Generate test data from the public LFW face dataset (research use only; not included in this repo):

```powershell
python -m pip install scikit-learn
python make_sample_dataset.py --people 10
python bulk_register.py
python -m streamlit run app.py
```

Then upload `sample_dataset/classroom/classroom_1.jpg` under **Take attendance** and compare with `sample_dataset/roster.csv`. No model training is needed: the app uses pretrained YOLOv8 and InsightFace models, and "registering" a student just stores face embeddings.
