import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import httpx
from fpdf import FPDF
from io import BytesIO
import textwrap

client = httpx.Client(verify=False)

llm = ChatOpenAI( 
    base_url="https://genailab.urg.in", 
    model="genailab-maas-gpt-5.4-mini", 
    api_key="sk-xxxxxxxxxx", 
    http_client=client 
    )

st.title("Travel Itinerary Planner")

# -----------------------------
# USER INPUTS
# -----------------------------
destination = st.text_input("Destination")
days = st.number_input("Number of days", min_value=1, max_value=10, value=3)

budget = st.selectbox("Budget", ["Low", "Medium", "High"])
pace = st.selectbox("Travel Pace", ["Relaxed", "Moderate", "Packed"])

interests = st.multiselect(
    "Interests",
    ["History", "Food", "Nature", "Shopping", "Adventure", "Culture"]
)

start_time = st.time_input("Start time", value=None)

# -----------------------------
# SIMPLE PLACE DATABASE (can expand later)
# -----------------------------
PLACE_DB = {
    "Paris": [
        "Eiffel Tower",
        "Louvre Museum",
        "Notre Dame",
        "Montmartre",
        "Seine River Cruise",
        "Luxembourg Gardens"
    ],
    "London": [
        "Big Ben",
        "London Eye",
        "British Museum",
        "Tower of London",
        "Buckingham Palace"
    ]
}

# -----------------------------
# GENERATE BUTTON
# -----------------------------
if st.button("Generate Itinerary"):

    if destination not in PLACE_DB:
        st.error("Destination not in database yet.")
    else:

        places = PLACE_DB[destination]

        # -----------------------------
        # BUILD PROMPT
        # -----------------------------
        prompt = f"""
You are an expert travel planner.

Create a {days}-day itinerary for {destination}.

User Preferences:
- Budget: {budget}
- Travel Pace: {pace}
- Interests: {interests}

Available Places:
{places}

Rules:
- Max 4–5 places per day
- Group nearby places logically
- Include time slots (morning, afternoon, evening)
- Include meals and breaks
- Keep itinerary realistic and not rushed

Output format:
Day-wise schedule with time slots and short reasoning.
"""

        # -----------------------------
        # CALL LLM
        # -----------------------------
        with st.spinner("Generating itinerary..."):
            response = llm.invoke(prompt)

        #itinerary = response.choices[0].message.content
        itinerary = (response.content)

        def create_pdf(itinerary_text):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", size=12)
            page_width = pdf.w - 20
            clean_text = itinerary.encode("latin-1", "ignore").decode("latin-1")
            
            for line in clean_text.split("\n"):
                if line.strip():
                    wrapped = textwrap.fill(line, width=80)
                    pdf.multi_cell(page_width, 10, wrapped)
                else:
                    pdf.ln(5)

            return bytes(pdf.output(dest="S"))
            #return pdf_bytes

        # -----------------------------
        # OUTPUT
        # -----------------------------
        st.subheader("Your Itinerary")
        st.write(itinerary)
        pdf_data = create_pdf(itinerary)

        #clean_text = re.sub(r'\*\*', '', itinerary)
        #clean_text = re.sub(r'#{1,6}\s*', '', clean_text)

        st.download_button(
            label="Download Itinerary PDF",
            data=pdf_data,
            file_name="travel_itinerary.pdf",
            mime="application/pdf")