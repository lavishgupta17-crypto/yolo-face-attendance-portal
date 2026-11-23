from datetime import date

import cv2
import streamlit as st

from attendance_core import (
    clear_database,
    decode_image,
    get_face_analyzer,
    recognize_attendance,
    register_student,
    registered_students,
    save_attendance_records,
    save_monthly_workbook,
)

st.set_page_config(page_title="YOLO Attendance Portal", page_icon="📷", layout="wide")
st.title("📷 YOLO Attendance Portal")
st.caption("Register students once, then upload one classroom photo to create the daily attendance sheet.")

@st.cache_resource(show_spinner="Loading face-recognition model…")
def face_analyzer():
    return get_face_analyzer()


tab_register, tab_attendance, tab_database = st.tabs(["Register students", "Take attendance", "Database"])

with tab_register:
    st.subheader("Add or update a student")
    st.info("Use 5–15 clear photos for each student. Each uploaded photo must contain exactly one face.")
    with st.form("registration"):
        left, right = st.columns(2)
        sap_id = left.text_input("SAP ID")
        name = right.text_input("Full name")
        photos = st.file_uploader("Face photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        submitted = st.form_submit_button("Save student")
    if submitted:
        try:
            with st.spinner("Creating face embeddings…"):
                saved, warnings = register_student(sap_id, name, photos, face_analyzer())
            st.success(f"Saved {saved} usable photo embedding(s) for {name}.")
            for warning in warnings:
                st.warning(warning)
        except Exception as error:
            st.error(str(error))

with tab_attendance:
    st.subheader("Upload the classroom photo")

    st.warning(
        "Review unknown or low-similarity results before treating "
        "this as an official attendance record."
    )

    attendance_date = st.date_input(
        "Attendance date",
        value=date.today()
    )

    classroom_photo = st.file_uploader(
        "Classroom photo",
        type=["jpg", "jpeg", "png"],
        key="class_photo"
    )

    threshold = st.slider(
        "Match confidence threshold",
        0.30,
        0.70,
        0.42,
        0.01,
        help="Increase this if you see false matches."
    )

    if classroom_photo and st.button(
        "Recognize and mark present",
        type="primary"
    ):
        try:
            image = decode_image(classroom_photo.getvalue())

            with st.spinner("Detecting and recognizing faces..."):
                attendance, annotated, unknown = recognize_attendance(
                    image,
                    face_analyzer(),
                    threshold,
                    attendance_date
                )

            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption="Green = matched, red = unknown"
            )

            st.metric(
                "Students marked present",
                len(attendance)
            )

            if unknown:
                st.warning(
                    f"{unknown} face(s) could not be matched."
                )

            if attendance.empty:
                st.error(
                    "No registered students were matched. "
                    "Try a clearer photo or lower the threshold carefully."
                )
            else:
                save_attendance_records(attendance)

                report_path = save_monthly_workbook(
                    attendance_date.year,
                    attendance_date.month
                )

                st.success(
                    f"Attendance was saved in {report_path.name}."
                )

                st.dataframe(
                    attendance,
                    use_container_width=True,
                    hide_index=True
                )

                report_name = attendance_date.strftime(
                    "attendance_%Y_%m.xlsx"
                )

                st.download_button(
                    "Download monthly Excel attendance",
                    report_path.read_bytes(),
                    file_name=report_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as error:
            st.error(str(error))

with tab_database:
    st.subheader("Registered students")
    students = registered_students()
    st.metric("Students", len(students))
    st.dataframe(students, use_container_width=True, hide_index=True)
    st.divider()
    confirm = st.checkbox("I understand this deletes all locally stored student photos and embeddings.")
    if st.button("Delete all database data", disabled=not confirm):
        clear_database()
        st.success("The local database has been deleted.")
        st.rerun()
