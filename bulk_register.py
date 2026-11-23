"""Register every student folder in sample_dataset/students (named <SAPID>_<Name>)."""
from pathlib import Path

import attendance_core as core


class FileUpload:
    """Mimics Streamlit's uploaded file for register_student()."""

    def __init__(self, path: Path):
        self.name, self._raw = path.name, path.read_bytes()

    def getvalue(self) -> bytes:
        return self._raw


def main(root: str = "sample_dataset/students") -> None:
    analyzer = core.get_face_analyzer()
    for folder in sorted(Path(root).iterdir()):
        if not folder.is_dir():
            continue
        sap, _, name = folder.name.partition("_")
        uploads = [FileUpload(p) for p in sorted(folder.glob("*.jpg"))]
        count, warnings = core.register_student(sap, name.replace("_", " "), uploads, analyzer)
        print(f"{sap} {name}: {count} photos registered", *warnings, sep="\n  " if warnings else "")
    print("Done. Start the app and open 'Take attendance'.")


if __name__ == "__main__":
    main()
