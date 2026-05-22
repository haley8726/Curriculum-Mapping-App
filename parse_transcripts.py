#!/usr/bin/env python3
"""Parse CSU transcript dump text into ML-ready tabular datasets.

This script extracts student-level, term-level, and course-level datapoints from
multi-transcript text dumps that follow the structure shown in
`ingest/Transcripts_all.txt`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

TRANSCRIPT_MARKER_RE = re.compile(r"^\s*Transcript\s+\d+(?:\s+of\s+\d+)?\s*$", re.IGNORECASE | re.MULTILINE)
HEADER_RE = re.compile(
    r"^Colorado State University Unofficial Transcript for\s+(?P<name>.+?)\s+\((?P<id>\d+)\)\s*$"
)
TERM_RE = re.compile(r"^(Spring|Summer|Fall|Winter)\s+(Semester|Session)?\s*(\d{4})$", re.IGNORECASE)
COURSE_CODE_RE = re.compile(r"^(?P<subject>[A-Z]+)-(?P<number>[0-9A-Z]+)(?:-(?P<section>[0-9A-Z]+))?$")

SEASON_ORDER = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}

METRIC_KEYS = {
    "Overall Credit Hours Earned": "overall_credit_hours_earned",
    "Colorado State University Credit Hours Earned": "csu_credit_hours_earned",
    "Colorado State University GPA Credit Hours": "csu_gpa_credit_hours",
    "Colorado State University Grade Points": "csu_grade_points",
    "Colorado State University Cumulative GPA": "csu_cumulative_gpa",
    "Transfer Credit Hours Earned": "transfer_credit_hours_earned",
}

GRADE_TO_POINTS = {
    "A+": 4.0,
    "A": 4.0,
    "A-": 3.7,
    "B+": 3.3,
    "B": 3.0,
    "B-": 2.7,
    "C+": 2.3,
    "C": 2.0,
    "C-": 1.7,
    "D+": 1.3,
    "D": 1.0,
    "D-": 0.7,
    "F": 0.0,
}

NON_GPA_GRADES = {
    "W",
    "I",
    "S",
    "U",
    "P",
    "NP",
    "NGC",
    "TA",
    "TB",
    "TB+",
    "TS",
}


@dataclass
class TranscriptRecord:
    student_name: Optional[str]
    student_id_raw: Optional[str]
    generated_at: Optional[str]
    curriculum_term: Optional[str]
    program_code: Optional[str]
    program_description: Optional[str]
    curriculum_level: Optional[str]
    programs: List[Dict[str, str]]
    metrics: Dict[str, Optional[float]]
    degrees_awarded: Dict[str, Optional[str]]
    academic_term_summary: List[Dict[str, str]]
    current_credit_courses: List[Dict[str, str]]
    completed_csu_courses: List[Dict[str, str]]
    transfer_courses: List[Dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse transcript dump into tabular files.")
    parser.add_argument(
        "--input",
        default="ingest/Transcripts_all.txt",
        help="Path to transcript dump text file.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where parsed files will be written.",
    )
    parser.add_argument(
        "--format",
        choices=["tsv", "csv"],
        default="tsv",
        help="Output delimiter format.",
    )
    parser.add_argument(
        "--include-names",
        action="store_true",
        help="Include student names in output. By default, only ID hashes are emitted.",
    )
    parser.add_argument(
        "--include-raw-id",
        action="store_true",
        help="Include un-hashed student IDs in output tables.",
    )
    return parser.parse_args()


def split_transcripts(blob: str) -> List[str]:
    matches = list(TRANSCRIPT_MARKER_RE.finditer(blob))
    if not matches:
        return [blob.strip()] if blob.strip() else []

    chunks: List[str] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(blob)
        chunk = blob[start:end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def parse_float(value: str) -> Optional[float]:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_term_key(term: str) -> Tuple[int, int]:
    term = (term or "").strip()
    match = TERM_RE.match(term)
    if not match:
        return (0, 0)
    season = match.group(1).title()
    year = int(match.group(3))
    return (year, SEASON_ORDER.get(season, 0))


def safe_hash(raw_id: Optional[str]) -> Optional[str]:
    if not raw_id:
        return None
    return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()


def parse_key_value_line(line: str) -> Optional[Tuple[str, str]]:
    if ":" not in line:
        return None
    left, right = line.split(":", 1)
    return left.strip(), right.strip()


def read_tabular_section(lines: Sequence[str], start_idx: int, expected_header: str) -> Tuple[List[Dict[str, str]], int]:
    rows: List[Dict[str, str]] = []

    if start_idx >= len(lines) or lines[start_idx].strip() != expected_header:
        return rows, start_idx

    idx = start_idx + 1
    if idx >= len(lines):
        return rows, idx

    headers = [h.strip() for h in lines[idx].split("\t")]
    idx += 1

    while idx < len(lines):
        raw = lines[idx]
        stripped = raw.strip()
        if not stripped:
            idx += 1
            continue
        if "\t" not in raw:
            break

        parts = [p.strip() for p in raw.split("\t")]
        if len(parts) < len(headers):
            parts = parts + [""] * (len(headers) - len(parts))
        elif len(parts) > len(headers):
            # Merge accidental split from tab noise into last column.
            parts = parts[: len(headers) - 1] + [" ".join(parts[len(headers) - 1 :])]

        row = {headers[i]: parts[i] for i in range(len(headers))}
        rows.append(row)
        idx += 1

    return rows, idx


def parse_transcript_chunk(chunk: str) -> TranscriptRecord:
    lines = [line.rstrip("\n") for line in chunk.splitlines() if line.strip()]

    student_name: Optional[str] = None
    student_id_raw: Optional[str] = None
    generated_at: Optional[str] = None
    curriculum_term: Optional[str] = None
    program_code: Optional[str] = None
    program_description: Optional[str] = None
    curriculum_level: Optional[str] = None
    programs: List[Dict[str, str]] = []
    metrics: Dict[str, Optional[float]] = {key: None for key in METRIC_KEYS.values()}
    degrees_awarded: Dict[str, Optional[str]] = {
        "degree_award_term": None,
        "degree_name": None,
        "degree_conferred_date": None,
        "degree_major": None,
        "degree_concentration": None,
        "degree_minor": None,
    }

    academic_term_summary: List[Dict[str, str]] = []
    current_credit_courses: List[Dict[str, str]] = []
    completed_csu_courses: List[Dict[str, str]] = []
    transfer_courses: List[Dict[str, str]] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        header_match = HEADER_RE.match(line)
        if header_match:
            student_name = header_match.group("name").strip()
            student_id_raw = header_match.group("id").strip()
            i += 1
            continue

        if generated_at is None and re.search(r"\d{1,2}:\d{2}:\d{2}\s+[AP]M$", line):
            generated_at = line
            i += 1
            continue

        if curriculum_term is None and line.endswith("Curriculum"):
            curriculum_term = line[: -len("Curriculum")].strip()
            i += 1
            continue

        parsed_kv = parse_key_value_line(line)
        if parsed_kv:
            key, value = parsed_kv
            if key == "Program Code":
                program_code = value
                i += 1
                continue
            if key == "Program Description":
                program_description = value
                i += 1
                continue
            if key == "Curriculum Level":
                curriculum_level = value
                i += 1
                continue
            if key in METRIC_KEYS:
                metrics[METRIC_KEYS[key]] = parse_float(value)
                i += 1
                continue

            if key == "Conferred":
                degrees_awarded["degree_conferred_date"] = value
                i += 1
                continue
            if key == "MAJOR":
                degrees_awarded["degree_major"] = value
                i += 1
                continue
            if key == "CONCENTRATION":
                degrees_awarded["degree_concentration"] = value
                i += 1
                continue
            if key == "MINOR":
                degrees_awarded["degree_minor"] = value
                i += 1
                continue

        if line == "Type\tDescription\tCode\tDepartment\tCollege":
            i += 1
            while i < len(lines) and "\t" in lines[i]:
                parts = [p.strip() for p in lines[i].split("\t")]
                if len(parts) >= 5 and parts[0] in {"MAJOR", "CONCENTRATION", "MINOR", "CERTIFICATE", "TRACK"}:
                    programs.append(
                        {
                            "type": parts[0],
                            "description": parts[1],
                            "code": parts[2],
                            "department": parts[3],
                            "college": parts[4],
                        }
                    )
                    i += 1
                    continue
                break
            continue

        if line == "Degrees Awarded":
            i += 1
            while i < len(lines):
                degree_line = lines[i].strip()
                if not degree_line:
                    i += 1
                    continue
                if degree_line in {
                    "Academic Term Summary",
                    "Current Credit Courses",
                    "Completed CSU Courses",
                    "Transfer Courses",
                }:
                    break
                kv = parse_key_value_line(degree_line)
                if kv:
                    key, value = kv
                    if key == "Conferred":
                        degrees_awarded["degree_conferred_date"] = value
                    elif key == "MAJOR":
                        degrees_awarded["degree_major"] = value
                    elif key == "CONCENTRATION":
                        degrees_awarded["degree_concentration"] = value
                    elif key == "MINOR":
                        degrees_awarded["degree_minor"] = value
                    i += 1
                    continue
                if " - " in degree_line and "Semester" in degree_line and not degree_line.startswith("Conferred"):
                    term, degree_name = degree_line.split(" - ", 1)
                    degrees_awarded["degree_award_term"] = term.strip()
                    degrees_awarded["degree_name"] = degree_name.strip()
                i += 1
            continue

        if line == "Academic Term Summary":
            table, next_i = read_tabular_section(lines, i, "Academic Term Summary")
            academic_term_summary = table
            i = next_i
            continue

        if line == "Current Credit Courses":
            table, next_i = read_tabular_section(lines, i, "Current Credit Courses")
            current_credit_courses = table
            i = next_i
            continue

        if line == "Completed CSU Courses":
            table, next_i = read_tabular_section(lines, i, "Completed CSU Courses")
            completed_csu_courses = table
            i = next_i
            continue

        if line == "Transfer Courses":
            table, next_i = read_tabular_section(lines, i, "Transfer Courses")
            transfer_courses = table
            i = next_i
            continue

        i += 1

    return TranscriptRecord(
        student_name=student_name,
        student_id_raw=student_id_raw,
        generated_at=generated_at,
        curriculum_term=curriculum_term,
        program_code=program_code,
        program_description=program_description,
        curriculum_level=curriculum_level,
        programs=programs,
        metrics=metrics,
        degrees_awarded=degrees_awarded,
        academic_term_summary=academic_term_summary,
        current_credit_courses=current_credit_courses,
        completed_csu_courses=completed_csu_courses,
        transfer_courses=transfer_courses,
    )


def normalize_grade(grade: str) -> str:
    return grade.strip().upper()


def grade_points(grade: str) -> Optional[float]:
    normalized = normalize_grade(grade)
    return GRADE_TO_POINTS.get(normalized)


def grade_counts_toward_gpa(grade: str) -> Optional[bool]:
    normalized = normalize_grade(grade)
    if not normalized:
        return None
    if normalized in GRADE_TO_POINTS:
        return True
    if normalized in NON_GPA_GRADES:
        return False
    return None


def parse_course_code(raw_course: str) -> Dict[str, Optional[str]]:
    course = raw_course.strip()
    match = COURSE_CODE_RE.match(course)
    if not match:
        return {
            "course_subject": None,
            "course_number": None,
            "course_section": None,
            "course_id": None,
        }

    subject = match.group("subject")
    number = match.group("number")
    section = match.group("section")
    return {
        "course_subject": subject,
        "course_number": number,
        "course_section": section,
        "course_id": f"{subject}-{number}",
    }


def latest_nonzero_term(term_rows: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    if not term_rows:
        return None

    ranked = sorted(term_rows, key=lambda r: parse_term_key(r.get("Term", "")), reverse=True)
    for row in ranked:
        term_gpa = parse_float(row.get("Term GPA", ""))
        gpa_hours = parse_float(row.get("GPA Hours", ""))
        if term_gpa and term_gpa > 0 and gpa_hours and gpa_hours > 0:
            return row

    return ranked[0] if ranked else None


def build_rows(
    records: List[TranscriptRecord], include_names: bool, include_raw_id: bool
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    student_rows: List[Dict[str, object]] = []
    term_rows_out: List[Dict[str, object]] = []
    course_rows: List[Dict[str, object]] = []

    for record in records:
        student_id_hash = safe_hash(record.student_id_raw)
        latest_term = latest_nonzero_term(record.academic_term_summary)
        has_degree = bool(record.degrees_awarded.get("degree_name"))

        major_codes = [p["code"] for p in record.programs if p.get("type") == "MAJOR"]
        concentration_codes = [p["code"] for p in record.programs if p.get("type") == "CONCENTRATION"]
        minor_codes = [p["code"] for p in record.programs if p.get("type") == "MINOR"]

        student_row: Dict[str, object] = {
            "student_id_hash": student_id_hash,
            "transcript_generated_at": record.generated_at,
            "curriculum_term": record.curriculum_term,
            "program_code": record.program_code,
            "program_description": record.program_description,
            "curriculum_level": record.curriculum_level,
            "major_codes": "|".join(major_codes),
            "concentration_codes": "|".join(concentration_codes),
            "minor_codes": "|".join(minor_codes),
            "overall_credit_hours_earned": record.metrics.get("overall_credit_hours_earned"),
            "csu_credit_hours_earned": record.metrics.get("csu_credit_hours_earned"),
            "csu_gpa_credit_hours": record.metrics.get("csu_gpa_credit_hours"),
            "csu_grade_points": record.metrics.get("csu_grade_points"),
            "current_gpa": record.metrics.get("csu_cumulative_gpa"),
            "final_gpa": record.metrics.get("csu_cumulative_gpa"),
            "transfer_credit_hours_earned": record.metrics.get("transfer_credit_hours_earned"),
            "has_degree_awarded": has_degree,
            "degree_award_term": record.degrees_awarded.get("degree_award_term"),
            "degree_name": record.degrees_awarded.get("degree_name"),
            "degree_conferred_date": record.degrees_awarded.get("degree_conferred_date"),
            "degree_major": record.degrees_awarded.get("degree_major"),
            "degree_concentration": record.degrees_awarded.get("degree_concentration"),
            "degree_minor": record.degrees_awarded.get("degree_minor"),
            "latest_term": latest_term.get("Term") if latest_term else None,
            "latest_term_gpa": parse_float(latest_term.get("Term GPA", "")) if latest_term else None,
            "latest_gpa_hours": parse_float(latest_term.get("GPA Hours", "")) if latest_term else None,
            "latest_hours_earned": parse_float(latest_term.get("Hours Earned", "")) if latest_term else None,
            "latest_term_standing": latest_term.get("End of Term Standing") if latest_term else None,
            "completed_csu_course_count": len(record.completed_csu_courses),
            "current_credit_course_count": len(record.current_credit_courses),
            "transfer_course_count": len(record.transfer_courses),
            "instructor_id": None,
        }

        if include_names:
            student_row["student_name"] = record.student_name
        if include_raw_id:
            student_row["student_id_raw"] = record.student_id_raw

        student_rows.append(student_row)

        term_lookup: Dict[str, Dict[str, str]] = {row.get("Term", ""): row for row in record.academic_term_summary}

        for term_row in record.academic_term_summary:
            term = term_row.get("Term", "")
            year, season_order = parse_term_key(term)
            season = term.split(" ", 1)[0] if term else None

            term_rows_out.append(
                {
                    "student_id_hash": student_id_hash,
                    "term": term,
                    "term_year": year if year else None,
                    "term_season": season,
                    "term_season_order": season_order if season_order else None,
                    "term_dates": term_row.get("Term Dates"),
                    "class_level": term_row.get("Class"),
                    "major": term_row.get("Major"),
                    "term_gpa": parse_float(term_row.get("Term GPA", "")),
                    "quality_points": parse_float(term_row.get("Quality Points", "")),
                    "gpa_hours": parse_float(term_row.get("GPA Hours", "")),
                    "hours_earned": parse_float(term_row.get("Hours Earned", "")),
                    "end_of_term_standing": term_row.get("End of Term Standing"),
                    "course_load": parse_float(term_row.get("GPA Hours", "")),
                    "current_gpa": record.metrics.get("csu_cumulative_gpa"),
                    "final_gpa": record.metrics.get("csu_cumulative_gpa"),
                    "instructor_id": None,
                }
            )
            if include_raw_id:
                term_rows_out[-1]["student_id_raw"] = record.student_id_raw

        def append_course_rows(rows: Iterable[Dict[str, str]], source: str) -> None:
            for row in rows:
                term = row.get("Term", "")
                term_meta = term_lookup.get(term, {})
                year, season_order = parse_term_key(term)
                season = term.split(" ", 1)[0] if term else None
                raw_course = row.get("Course", "")
                parsed_course = parse_course_code(raw_course)
                grade_raw = row.get("Grade", "")

                course_rows.append(
                    {
                        "student_id_hash": student_id_hash,
                        "record_source": source,
                        "term": term,
                        "term_year": year if year else None,
                        "term_season": season,
                        "term_season_order": season_order if season_order else None,
                        "course_raw": raw_course,
                        "course_id": parsed_course["course_id"],
                        "course_subject": parsed_course["course_subject"],
                        "course_number": parsed_course["course_number"],
                        "course_section": parsed_course["course_section"],
                        "course_title": row.get("Title"),
                        "course_credits": parse_float(row.get("Credits", "")),
                        "course_grade": grade_raw if grade_raw else None,
                        "grade_points_4_scale": grade_points(grade_raw) if grade_raw else None,
                        "counts_toward_gpa": grade_counts_toward_gpa(grade_raw) if grade_raw else None,
                        "level": row.get("Level"),
                        "comments": row.get("Comments"),
                        "institution": row.get("Institution"),
                        "term_class_level": term_meta.get("Class"),
                        "term_major": term_meta.get("Major"),
                        "term_gpa": parse_float(term_meta.get("Term GPA", "")),
                        "term_quality_points": parse_float(term_meta.get("Quality Points", "")),
                        "term_gpa_hours": parse_float(term_meta.get("GPA Hours", "")),
                        "term_hours_earned": parse_float(term_meta.get("Hours Earned", "")),
                        "end_of_term_standing": term_meta.get("End of Term Standing"),
                        "course_load": parse_float(term_meta.get("GPA Hours", "")),
                        "current_gpa": record.metrics.get("csu_cumulative_gpa"),
                        "final_gpa": record.metrics.get("csu_cumulative_gpa"),
                        "program_code": record.program_code,
                        "curriculum_level": record.curriculum_level,
                        "instructor_id": None,
                    }
                )
                if include_raw_id:
                    course_rows[-1]["student_id_raw"] = record.student_id_raw

        append_course_rows(record.current_credit_courses, "current_credit")
        append_course_rows(record.completed_csu_courses, "completed_csu")
        append_course_rows(record.transfer_courses, "transfer")

    return student_rows, term_rows_out, course_rows


def write_table(rows: List[Dict[str, object]], out_path: Path, dialect: str) -> None:
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return

    delimiter = "\t" if dialect == "tsv" else ","
    headers = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    blob = input_path.read_text(encoding="utf-8")
    chunks = split_transcripts(blob)
    records = [parse_transcript_chunk(chunk) for chunk in chunks]

    student_rows, term_rows, course_rows = build_rows(
        records,
        include_names=args.include_names,
        include_raw_id=args.include_raw_id,
    )

    ext = "tsv" if args.format == "tsv" else "csv"
    write_table(student_rows, output_dir / f"student_datapoints.{ext}", args.format)
    write_table(term_rows, output_dir / f"term_datapoints.{ext}", args.format)
    write_table(course_rows, output_dir / f"course_datapoints.{ext}", args.format)

    print(f"Parsed transcripts: {len(records)}")
    print(f"Student rows: {len(student_rows)}")
    print(f"Term rows: {len(term_rows)}")
    print(f"Course rows: {len(course_rows)}")
    print(f"Output format: {args.format}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
