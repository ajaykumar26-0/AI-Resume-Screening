import streamlit as st
from resume_parser import extract_text_from_pdf
from preprocess import preprocess_text
from model import recommend_top_jobs, extract_skills

st.set_page_config(page_title="AI Resume Screening System")

st.title("📄 AI Resume Screening & Job Recommendation System")

uploaded_file = st.file_uploader(
    "Upload your Resume (PDF)", type=["pdf"]
)

if uploaded_file is not None:
    if st.button("Analyze Resume"):
        with st.spinner("Analyzing resume..."):
            raw_text = extract_text_from_pdf(uploaded_file)
            clean_text = preprocess_text(raw_text)

            top_jobs = recommend_top_jobs(
                clean_text, "jobs.csv"
            )
            skills = extract_skills(clean_text)

        st.success("Analysis Complete ✅")

        st.subheader("🧑‍💼 Recommended Job Roles")
        for _, row in top_jobs.iterrows():
            st.write(
                f"**{row['Job Role']}** — {row['Match Score']:.2f}% match"
            )

        st.subheader("🛠️ Extracted Skills")
        if skills:
            st.write(", ".join(skills))
        else:
            st.write("No skills detected")
