import pandas as pd

print("SCRIPT IS RUNNING")

# -----------------------------------
# LOAD FILES
# -----------------------------------
courses = pd.read_csv("course_datapoints.tsv", sep="\t", low_memory=False)
terms = pd.read_csv("term_datapoints.tsv", sep="\t")
students = pd.read_csv("student_datapoints.tsv", sep="\t")

# Clean column names
courses.columns = courses.columns.str.strip()
terms.columns = terms.columns.str.strip()
students.columns = students.columns.str.strip()

print("STUDENT COLUMNS:", students.columns.tolist())

# -----------------------------------
# CLEAN ID FIELDS ✅ (CRITICAL FIX)
# -----------------------------------
courses["student_id_hash"] = courses["student_id_hash"].astype(str).str.strip()
terms["student_id_hash"] = terms["student_id_hash"].astype(str).str.strip()
students["student_id_hash"] = students["student_id_hash"].astype(str).str.strip()

# -----------------------------------
# MAJOR MAPPING ✅
# -----------------------------------
program_to_major = {
    'BIOM-EPHZ-BS': 'Environmental and Public Health',
    'BIOM-APHZ-BS': 'Anatomy and Physiology',
    'BIOM-MIDZ-BS': 'Microbiology and Infectious Disease',
    'NERO-BCNZ-BS': 'Behavioral and Cognitive Neuroscience',
    'NERO-CMNZ-BS': 'Cell and Molecular Neuroscience',
    'NERO-BS': 'Neuroscience',
    'HAES-EXSZ-BS': 'Exercise Science',
    'HAES-BS': 'Exercise Science',
    'HAES-HPRZ-BS': 'Health Promotion'
}

# -----------------------------------
# CLEAN COURSE DATA
# -----------------------------------
courses = courses[
    (courses["record_source"] == "completed_csu") &
    (courses["counts_toward_gpa"] == True) &
    (courses["course_grade"].notna())
]
# -----------------------------------
# EXTRACT COURSE SECTION TYPE ✅
# -----------------------------------

# Extract section suffix (last part after underscore)
courses["section_code"] = (
    courses["course_raw"]
    .astype(str)
    .str.split("-")
    .str[-1]     # ✅ last part
    .str.strip()
    .str.upper()
)


courses = courses[courses["course_grade"] != "NGC"]

courses["course_key"] = courses["course_subject"] + "-" + courses["course_number"].astype(str)

def classify_section(section):
    if pd.isna(section):
        return "Other"
    section = str(section)
    if section.startswith("L"):
        return "Lab"
    elif section.startswith("R"):
        return "Recitation"
    elif section.isdigit():
        return "Lecture"
    else:
        return "Other"

courses["course_component"] = courses["section_code"].apply(classify_section)

# -----------------------------------
# TERM PROCESSING
# -----------------------------------
terms["term_order"] = terms["term_year"] * 10 + terms["term_season_order"]

terms = terms.sort_values(["student_id_hash", "term_order"])

terms["csu_credits_at_time"] = terms.groupby("student_id_hash")["hours_earned"].cumsum()

# -----------------------------------
# TRANSFER HANDLING
# -----------------------------------
transfer_lookup = students.set_index("student_id_hash")[
    "transfer_credit_hours_earned"
].fillna(0)

terms["transfer_credits"] = terms["student_id_hash"].map(transfer_lookup).fillna(0)

terms["is_first_term"] = terms.groupby("student_id_hash")["term_order"].transform("min") == terms["term_order"]

terms["all_credits_term"] = terms["hours_earned"]
terms.loc[terms["is_first_term"], "all_credits_term"] += terms["transfer_credits"]

terms["all_credits_at_time"] = terms.groupby("student_id_hash")["all_credits_term"].cumsum()

# -----------------------------------
# MERGE COURSES + TERMS
# -----------------------------------
df = courses.merge(
    terms[[
        "student_id_hash",
        "term",
        "term_order",
        "class_level",
        "major",
        "csu_credits_at_time",
        "all_credits_at_time"
    ]],
    on=["student_id_hash", "term"],
    how="left"
)

# -----------------------------------
# MERGE PROGRAM CODE ✅ (SAFE VERSION)
# -----------------------------------
df = df.merge(
    students[["student_id_hash", "program_code"]],
    on="student_id_hash",
    how="left",
    suffixes=("", "_student")
)
# -----------------------------------
# FIX term_year duplication ✅
# -----------------------------------

# Fix program_code if duplicated
if "program_code_student" in df.columns:
    df["program_code"] = df["program_code_student"]
    df = df.drop(columns=["program_code_student"])

print("DF COLUMNS AFTER MERGE:", df.columns.tolist())
print("Missing program_code:", df["program_code"].isna().sum())

# -----------------------------------
# REMOVE DUPLICATE COLUMNS ✅
# -----------------------------------
df = df.loc[:, ~df.columns.duplicated()]

# -----------------------------------
# CLASS LEVEL FIX ✅
# -----------------------------------
if "class_level" in df.columns:
    df["term_class_level"] = df["class_level"]

cols_to_drop = [col for col in df.columns if "class_level" in col and col != "term_class_level"]
df = df.drop(columns=cols_to_drop)

# -----------------------------------
# CLEAN COLUMN NAMES
# -----------------------------------
df = df.rename(columns={
    "major": "major_at_time"
})

# Remove incomplete rows
df = df[df["term_class_level"].notna()]

# -----------------------------------
# MAJOR GROUPING ✅
# -----------------------------------
df["major_group"] = df["program_code"].map(program_to_major)
df["major_group"] = df["major_group"].fillna("Other")

allowed_majors = list(program_to_major.values()) + ["Other"]
df = df[df["major_group"].isin(allowed_majors)]

# -----------------------------------
# REPEAT ATTEMPTS ✅
# -----------------------------------
df = df.sort_values(["student_id_hash", "course_key", "term_order"])

df["attempt_number"] = df.groupby(
    ["student_id_hash", "course_key"]
).cumcount() + 1

df["is_repeat"] = df["attempt_number"] > 1

# -----------------------------------
# SEMESTER CREDIT LOAD ✅
# -----------------------------------
# ✅ TRUE semester credit load (sum of enrolled credits)
df["term_credit_load"] = df.groupby(
    ["student_id_hash", "term"]
)["course_credits"].transform("sum")

# -----------------------------------
# FINAL OUTPUT
# -----------------------------------
final_df = df[[
    "student_id_hash",
    "term",
    "term_order",
    "course_key",
    "course_grade",
    "grade_points_4_scale",
    "attempt_number",
    "is_repeat",
    "term_class_level",
    "major_at_time",
    "major_group",
    "term_credit_load",
    "csu_credits_at_time",
    "all_credits_at_time",
    "course_component",
]].copy()

final_df = final_df.rename(columns={
    "grade_points_4_scale": "grade_points"
})

final_df["term_season"] = final_df["term"].str.extract(r'(Spring|Summer|Fall)')

# -----------------------------------
# SAVE FILE
# -----------------------------------
final_df.to_csv("student_course_trajectory.csv", index=False)

print("✅ student_course_trajectory.csv created!")
print("Unique major groups:", sorted(final_df["major_group"].unique()))
print("Missing class level:", final_df["term_class_level"].isna().sum())