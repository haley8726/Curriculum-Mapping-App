import streamlit as st
import pandas as pd
import plotly.express as px
import datetime

# ✅ Initialize once
if "log_messages" not in st.session_state:
    st.session_state.log_messages = []

# ✅ Reset logs each run (optional but recommended)
st.session_state.log_messages = []

# ✅ Log function
def log(section, message, level="INFO"):
    time = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.log_messages.append(
        f"[{time}] [{level}] [{section}] {message}"
    )

st.set_page_config(layout="wide")
# ------------------------------
# LOAD DATA ✅
# ------------------------------
df = pd.read_csv("student_course_trajectory.csv", low_memory=False)
df.columns = df.columns.str.strip()

log("LOAD", "Dataset initialized")
log("LOAD", f"Rows loaded: {len(df)}")
# --------------------------------------------------
# ✅ CLEAN CLASS LEVELS (GLOBAL FIX)
# --------------------------------------------------
df["term_class_level"] = df["term_class_level"].replace({
    "Senior - Second Bachelor": "Senior"
})

# ------------------------------
# MAIN DASHBOARD
# ------------------------------
st.title("Course Timing & Grade Explorer")

st.markdown("""
This dashboard explores **when students take courses and how they perform**, 
broken down by major. Use the filters below to explore patterns across programs.
""")

# ------------------------------
# FILTERS
# ------------------------------

# ---- TIME FILTER ----
df["term_season"] = df["term"].str.extract(r'(Spring|Summer|Fall)')
df["term_year"] = df["term"].str.extract(r'(\d{4})').astype(int)
df["term_label"] = df["term_season"] + " " + df["term_year"].astype(str)

term_options = (
    df[["term_label", "term_order"]]
    .drop_duplicates()
    .sort_values("term_order")
)

term_list = term_options["term_label"].tolist()

start_term = st.selectbox("Start Term", term_list, index=0)
end_term = st.selectbox("End Term", term_list, index=len(term_list)-1)


start_value = term_options[term_options["term_label"] == start_term]["term_order"].values[0]
end_value = term_options[term_options["term_label"] == end_term]["term_order"].values[0]

df = df[
    (df["term_order"] >= start_value) &
    (df["term_order"] <= end_value)
]

# ------------------------------
# COURSE COMPONENT FILTER ✅
# ------------------------------
section_filter = st.selectbox(
    "Course Component",
    ["All", "Lecture", "Lab", "Recitation"]
)


if section_filter != "All":
    df = df[df["course_component"] == section_filter]

   
log("FILTER", f"Term range: {start_term} → {end_term}")


# ------------------------------
# COLOR MAP ✅ (matches your majors)
# ------------------------------
major_colors = {
    'Environmental and Public Health': "#039C24",
    'Anatomy and Physiology': "#375af5",
    'Microbiology and Infectious Disease': "#f87306",
    'Behavioral and Cognitive Neuroscience': "#ee99cd",
    'Cell and Molecular Neuroscience': "#64ecda",
    'Neuroscience': "#f8e856",
    'Exercise Science': "#ac1c1c",
    'Health Promotion': "#6f2083",
    'Other': '#999999'
}
grade_colors = {
    "A+": "#1b7837", "A": "#1b7837", "A-": "#5aae61",
    "B+": "#a6dba0", "B": "#d9f0d3", "B-": "#fee08b",
    "C+": "#fdae61", "C": "#f46d43", "C-": "#d73027",
    "D": "#a50026", "F": "#67001f", "W": "#bdbdbd"
}
# ------------------------------
# GRADE ORDER
# ------------------------------
grade_order = [
    "A+", "A", "A-",
    "B+", "B", "B-",
    "C+", "C", "C-",
    "D", "F", "W"
]

df["course_grade"] = pd.Categorical(
    df["course_grade"],
    categories=grade_order,
    ordered=True
)

# ------------------------------
# AGGREGATE ✅ (FIXED)
# ------------------------------
agg = (
    df.groupby([
        "course_key",
        "major_group",   # ✅ FIXED
        "term_class_level",
        "term_season",
        "course_grade"
    ])
    .size()
    .reset_index(name="total_enrollments")
)

# ------------------------------
# CLASS LEVEL ORDER
# ------------------------------
level_order = ["Freshman", "Sophomore", "Junior", "Senior"]

df["term_class_level"] = pd.Categorical(
    df["term_class_level"],
    categories=level_order,
    ordered=True
)

agg["term_class_level"] = pd.Categorical(
    agg["term_class_level"],
    categories=level_order,
    ordered=True
)

# ------------------------------
# COURSE SELECTOR
# ------------------------------
course_list = sorted(agg["course_key"].unique())

course_search = st.text_input("Search for a course (e.g., BMS-300)")

filtered_courses = [
    c for c in course_list if course_search.upper() in c
] if course_search else course_list

selected_course = st.selectbox("Select a course", filtered_courses)

# Start from df (not agg!)
filtered_df = df[df["course_key"] == selected_course]


agg = (
    filtered_df.groupby([
        "course_key",
        "major_group",
        "term_class_level",
        "term_season",
        "course_grade"
    ])
    .size()
    .reset_index(name="total_enrollments")
)
log("SELECT", f"Course selected: {selected_course}")
# ============================================================
# ✅ GRAPH 1: TIMING BY MAJOR
# ============================================================

st.markdown("## When Students Take This Course")

st.markdown(
    "This chart shows **when students take the selected course across majors**, "
    "broken down by class level (Freshman through Senior). "
    "Each bar represents the **percentage of enrollments within a major**, "
    "and labels show the number of students (n). "
    "This helps identify when different majors typically complete the course."
)

log("GRAPH 1", f"Building timing distribution (rows={len(filtered_df)})")

timing_df = filtered_df.groupby(
    ["major_group", "term_class_level"],
    observed=False
).size().reset_index(name="total_enrollments")
timing_df["label"] = timing_df["total_enrollments"].apply(
    lambda x: f"n={x}" if x > 0 else ""
)

# ✅ Remove empty class levels
timing_df = timing_df[timing_df["total_enrollments"] > 0]

if timing_df.empty:
    st.warning("No data available for selected filters.")
else:
    timing_df["total_by_major"] = timing_df.groupby("major_group")["total_enrollments"].transform("sum")
    timing_df["percent"] = timing_df["total_enrollments"] / timing_df["total_by_major"] * 100

    fig1 = px.bar(
        timing_df,
        x="term_class_level",
        y="percent",
        color="major_group",
        text="label",
        barmode="group",
        category_orders={"term_class_level": level_order},
        color_discrete_map=major_colors
    )

    fig1.update_traces(textposition='outside')

    st.plotly_chart(fig1, use_container_width=True)
st.write("Graph 1 total:", timing_df["total_enrollments"].sum())

# ============================================================
# ✅ GRAPH 2: GRADE DISTRIBUTION
# ============================================================
st.markdown("---")
st.markdown("## How Students Perform")

st.markdown(
    "This chart shows the **distribution of grades by major** for the selected course. "
    "Percentages are calculated within each major, allowing comparison of performance patterns. "
    "Use this to see whether certain majors tend to perform better or worse in this course."
)
log("GRAPH 2", f"Building grade distribution (rows={len(filtered_df)})")
grade_df = filtered_df.groupby(
    ["major_group", "course_grade"],
    observed=False
).size().reset_index(name="total_enrollments")
# ✅ Remove empty grades
grade_df = grade_df[grade_df["total_enrollments"] > 0]
# ✅ Only keep grades that exist, but preserve correct order
filtered_grade_order = [
    g for g in grade_order if g in grade_df["course_grade"].unique()
]
grade_df["label"] = grade_df["total_enrollments"].apply(
    lambda x: f"n={x}" if x > 0 else ""
)

if grade_df.empty:
    st.warning("No data available for selected filters.")
else:
    grade_df["total_by_major"] = grade_df.groupby("major_group")["total_enrollments"].transform("sum")
    grade_df["percent"] = grade_df["total_enrollments"] / grade_df["total_by_major"] * 100

    fig2 = px.bar(
        grade_df,
        x="course_grade",
        y="percent",
        color="major_group",
        text="label",
        barmode="group",
        category_orders={"course_grade": filtered_grade_order},
        color_discrete_map=major_colors,
    )

    fig2.update_traces(textposition='outside')

    st.plotly_chart(fig2, use_container_width=True)
    st.write("Graph 2 total:", grade_df["total_enrollments"].sum())
# ============================================================
# ✅ GRAPH 3: TIMING VS PERFORMANCE
# ============================================================
st.markdown("---")
st.markdown("## How Timing Relates to Performance")

st.markdown(
    "This chart shows how **student performance varies depending on when they take the course**. "
    "It compares grade distributions across class levels, helping identify whether taking the course earlier "
    "or later in a program is associated with different outcomes."
)

selected_major = st.selectbox(
    "Select a Major",
    ["All"] + sorted(filtered_df["major_group"].unique())
)

filtered_major_df = filtered_df.copy()

# ✅ Create attempt type instead of filtering
filtered_major_df["attempt_type"] = filtered_major_df["attempt_number"].apply(
    lambda x: "First Attempt" if x == 1 else "Repeat"
)
log("GRAPH 3", f"Splitting attempts (rows={len(filtered_major_df)})")

if selected_major != "All":
    filtered_major_df = filtered_major_df[
        filtered_major_df["major_group"] == selected_major
    ]

timing_grade_df = filtered_major_df.groupby(
    ["term_class_level", "course_grade", "attempt_type"],
    observed=False
).size().reset_index(name="total_enrollments")
# ✅ Add clean labels (hide zeros)
timing_grade_df["label"] = timing_grade_df["total_enrollments"].apply(
    lambda x: f"n={x}" if x > 0 else ""
)
timing_grade_df = timing_grade_df[timing_grade_df["total_enrollments"] > 0]

# ✅ Enforce correct class level order (Graph 3)
level_order = ["Freshman", "Sophomore", "Junior", "Senior"]

timing_grade_df["term_class_level"] = pd.Categorical(
    timing_grade_df["term_class_level"],
    categories=level_order,
    ordered=True
)


if timing_grade_df.empty:
    st.warning("No data available for selected filters.")
else:
    timing_grade_df["total_per_level"] = timing_grade_df.groupby(
        "term_class_level"
    )["total_enrollments"].transform("sum")

    timing_grade_df["percent"] = timing_grade_df["total_enrollments"] / timing_grade_df["total_per_level"] * 100

    fig3 = px.bar(
    timing_grade_df,
    x="term_class_level",
    y="percent",
    color="course_grade",
    text="label",
    barmode="group",
    facet_col="attempt_type",   # ✅ THIS IS THE KEY CHANGE
    category_orders={
    "term_class_level": level_order,   # ✅ THIS FIXES YOUR AXIS
    "course_grade": grade_order,
    "attempt_type": ["First Attempt", "Repeat"]},
    color_discrete_map=grade_colors
)

fig3.update_traces(textposition='outside')
st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# ✅ GRAPH 4: CREDIT LOAD
# ============================================================

st.markdown("---")
st.markdown("## How Course Load Relates to Performance")

st.markdown(
    "This chart shows how **grades vary based on a student’s semester credit load**. "
    "Each group represents a range of credits completed during a term, and bars show "
    "the percentage of grades earned. "
    "This helps evaluate whether heavier or lighter course loads are associated with performance."
)

# ✅ Major filter
selected_major_load = st.selectbox(
    "Select Major",
    ["All"] + sorted(df["major_group"].unique())
)

# ✅ Create working dataframe
load_df = filtered_df.copy()

# ✅ Filter to selected course
load_df = load_df[load_df["course_key"] == selected_course]
log("GRAPH 4", f"Computing credit load bins (rows={len(load_df)})")
# ✅ Filter by major
if selected_major_load != "All":
    load_df = load_df[load_df["major_group"] == selected_major_load]

# ✅ Term filter UI FIRST
with st.expander("Filter Semesters", expanded=False):

    all_terms = sorted(df["term_label"].unique())

    selected_terms_for_load = st.multiselect(
        "Included Semesters",
        options=all_terms,
        default=all_terms
    )

# ✅ Now apply term filter (ONLY ONCE)
load_df = load_df[load_df["term_label"].isin(selected_terms_for_load)]

st.caption(f"{len(selected_terms_for_load)} of {len(all_terms)} semesters selected")

load_df = load_df[load_df["term_credit_load"].notna()]
# ✅ Create credit load bins
load_df["credit_load_bin"] = pd.cut(
    load_df["term_credit_load"],
    bins=[0, 6, 12, 15, 18, 30],
    labels=["0-6", "7-12", "13-15", "16-18", "19+"]
)

# ✅ Aggregate AFTER filtering
credit_perf_df = load_df.groupby(
    ["credit_load_bin", "course_grade"]
).size().reset_index(name="count")

credit_perf_df["label"] = credit_perf_df["count"].apply(
    lambda x: f"n={x}" if x > 0 else ""
)
# ✅ Handle empty case
if credit_perf_df.empty:
    st.warning("No data available for selected filters.")
else:
    credit_perf_df["total"] = credit_perf_df.groupby("credit_load_bin")["count"].transform("sum")
    credit_perf_df["percent"] = credit_perf_df["count"] / credit_perf_df["total"] * 100

    # ✅ Grade colors
    grade_colors = {
        "A+": "#1b7837", "A": "#1b7837", "A-": "#5aae61",
        "B+": "#a6dba0", "B": "#d9f0d3", "B-": "#fee08b",
        "C+": "#fdae61", "C": "#f46d43", "C-": "#d73027",
        "D": "#a50026", "F": "#67001f", "W": "#bdbdbd"
    }
    credit_order = ["0-6", "7-12", "13-15", "16-18", "19+"]
    fig4 = px.bar(
    credit_perf_df,
    x="credit_load_bin",
    y="percent",
    color="course_grade",
    text="label",
    barmode="group",
    category_orders={
        "credit_load_bin": credit_order,   # ✅ FIX HERE
        "course_grade": grade_order
    },
    color_discrete_map=grade_colors
)

    fig4.update_traces(textposition='outside')

    st.plotly_chart(fig4, use_container_width=True)
with st.sidebar:
    st.markdown("### 🖥️ Debug Terminal")

    log_text = "\n".join(reversed(st.session_state.log_messages[-40:]))

    st.text_area(
        "Execution Log",
        value=log_text,
        height=400
    )