import streamlit as st
import pandas as pd
import joblib




st.set_page_config(
    page_title="DreamHome | House Price Predictor",
    page_icon="🏡",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("house_prices.pk1")


model = load_model()

 Exact features used while training the model
FEATURES = list(model.feature_names_in_)



st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(
        135deg,
        #dff7e8 0%,
        #f4fff8 45%,
        #e8f7ff 100%
    );
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Header */
.hero {
    background: linear-gradient(
        135deg,
        #075e45,
        #0b8060,
        #16a085
    );
    padding: 35px;
    border-radius: 25px;
    color: white;
    box-shadow: 0 12px 35px rgba(0,0,0,0.12);
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.hero p {
    font-size: 17px;
    opacity: 0.92;
}

/* House illustration */
.house-card {
    background: linear-gradient(
        180deg,
        #9ee7ff 0%,
        #dff8ff 62%,
        #78c978 63%,
        #4da95a 100%
    );
    height: 340px;
    border-radius: 25px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 12px 30px rgba(0,0,0,0.12);
}

/* Sun */
.sun {
    position: absolute;
    right: 55px;
    top: 35px;
    width: 70px;
    height: 70px;
    background: #ffd95a;
    border-radius: 50%;
    box-shadow: 0 0 35px rgba(255,217,90,0.7);
}

/* House */
.house {
    position: absolute;
    width: 230px;
    height: 145px;
    background: #fffaf0;
    left: 50%;
    bottom: 55px;
    transform: translateX(-50%);
    border-radius: 8px;
    box-shadow: 0 12px 20px rgba(0,0,0,0.15);
}

/* Roof */
.roof {
    position: absolute;
    width: 0;
    height: 0;
    border-left: 145px solid transparent;
    border-right: 145px solid transparent;
    border-bottom: 105px solid #8c4a35;
    left: 50%;
    bottom: 165px;
    transform: translateX(-50%);
}

/* Door */
.door {
    position: absolute;
    width: 48px;
    height: 78px;
    background: #7b4b32;
    bottom: 0;
    left: 91px;
    border-radius: 5px 5px 0 0;
}

/* Door knob */
.knob {
    position: absolute;
    width: 7px;
    height: 7px;
    background: #ffd45c;
    border-radius: 50%;
    right: 7px;
    top: 40px;
}

/* Windows */
.window {
    position: absolute;
    width: 48px;
    height: 45px;
    background: #8edff5;
    border: 5px solid #694536;
    top: 38px;
}

.window.left {
    left: 22px;
}

.window.right {
    right: 22px;
}

/* Fairy lights */
.lights {
    position: absolute;
    width: 190px;
    height: 2px;
    background: #4c342d;
    left: 50%;
    bottom: 165px;
    transform: translateX(-50%) rotate(2deg);
}

.light {
    position: absolute;
    width: 9px;
    height: 9px;
    background: #fff4a3;
    border-radius: 50%;
    box-shadow: 0 0 12px #fff4a3;
}

/* Trees */
.tree {
    position: absolute;
    bottom: 45px;
    width: 70px;
    height: 130px;
}

.tree.left {
    left: 35px;
}

.tree.right {
    right: 35px;
}

.tree-top {
    position: absolute;
    width: 70px;
    height: 90px;
    background: #237a42;
    border-radius: 50%;
    top: 0;
}

.tree-trunk {
    position: absolute;
    width: 18px;
    height: 65px;
    background: #70452e;
    bottom: 0;
    left: 26px;
}

/* Cards */
.info-card {
    background: rgba(255,255,255,0.86);
    padding: 22px;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.section-title {
    color: #075e45;
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 15px;
}

/* Button */
.stButton > button,
.stFormSubmitButton > button {
    width: 100%;
    border-radius: 14px;
    border: none;
    background: linear-gradient(
        90deg,
        #087f5b,
        #15a77d
    );
    color: white;
    font-size: 17px;
    font-weight: 600;
    padding: 13px;
    box-shadow: 0 8px 18px rgba(8,127,91,0.25);
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(
        90deg,
        #056c4d,
        #0c8c69
    );
}

/* Prediction result */
.result-box {
    background: linear-gradient(
        135deg,
        #075e45,
        #14a77d
    );
    color: white;
    padding: 30px;
    border-radius: 22px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.15);
}

.result-box h2 {
    font-size: 18px;
    margin-bottom: 5px;
}

.result-box h1 {
    font-size: 38px;
    margin: 0;
}

.footer {
    text-align: center;
    color: #557066;
    margin-top: 35px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------

st.markdown("""
<div class="hero">

<h1>🏡 DreamHome</h1>

<p>
Smart House Price Prediction using Machine Learning
</p>

</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# HOUSE ILLUSTRATION
# ---------------------------------------------------------

st.markdown("""
<div class="house-card">

<div class="sun"></div>

<div class="tree left">
    <div class="tree-top"></div>
    <div class="tree-trunk"></div>
</div>

<div class="tree right">
    <div class="tree-top"></div>
    <div class="tree-trunk"></div>
</div>

<div class="roof"></div>

<div class="house">

    <div class="window left"></div>
    <div class="window right"></div>

    <div class="door">
        <div class="knob"></div>
    </div>

</div>

<div class="lights">

<div class="light" style="left:10px;"></div>
<div class="light" style="left:40px;"></div>
<div class="light" style="left:70px;"></div>
<div class="light" style="left:100px;"></div>
<div class="light" style="left:130px;"></div>
<div class="light" style="left:160px;"></div>

</div>

</div>
""", unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------

with st.form("house_form"):

    st.markdown(
        '<div class="section-title">🏠 Property Details</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        carpet_area = st.number_input(
            "📐 Carpet Area",
            min_value=0.0,
            value=1000.0,
            step=10.0
        )

        bathroom = st.number_input(
            "🚿 Bathroom",
            min_value=0.0,
            value=2.0,
            step=1.0
        )

    with col2:

        balcony = st.number_input(
            "🌿 Balcony",
            min_value=0.0,
            value=1.0,
            step=1.0
        )

        floor_number = st.number_input(
            "🏢 Floor Number",
            min_value=0.0,
            value=1.0,
            step=1.0
        )

    with col3:

        total_floors = st.number_input(
            "🏙️ Total Floors",
            min_value=0.0,
            value=5.0,
            step=1.0
        )

        location = st.selectbox(
            "📍 Location",
            [
                "ahmedabad",
                "allahabad",
                "aurangabad",
                "badlapur",
                "bangalore",
                "bhiwadi",
                "bhopal",
                "bhubaneswar",
                "chandigarh",
                "chennai",
                "coimbatore",
                "dehradun",
                "durgapur",
                "ernakulam",
                "faridabad",
                "ghaziabad",
                "goa",
                "greater-noida",
                "guntur",
                "gurgaon",
                "guwahati",
                "gwalior",
                "haridwar",
                "hyderabad",
                "indore",
                "jabalpur",
                "jaipur",
                "jamshedpur",
                "kalyan",
                "kanpur",
                "kochi",
                "kolkata",
                "lucknow",
                "ludhiana",
                "mangalore",
                "mohali",
                "mumbai",
                "mysore",
                "nagpur",
                "nashik",
                "navi-mumbai",
                "new-delhi",
                "noida",
                "other",
                "palghar",
                "panchkula",
                "patna",
                "pune",
                "raipur",
                "ranchi",
                "siliguri",
                "sonipat",
                "surat",
                "thane",
                "thrissur",
                "trichy",
                "trivandrum",
                "udaipur",
                "vadodara",
                "vapi",
                "varanasi",
                "vijayawada",
                "visakhapatnam",
                "zirakpur"
            ]
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)

    with col4:

        furnishing = st.selectbox(
            "🛋️ Furnishing",
            [
                "Unfurnished",
                "Semi-Furnished"
            ]
        )

    with col5:

        transaction = st.selectbox(
            "💳 Transaction",
            [
                "Other",
                "Rent/Lease",
                "Resale"
            ]
        )

    with col6:

        ownership = st.selectbox(
            "📄 Ownership",
            [
                "Freehold",
                "Leasehold",
                "Power Of Attorney"
            ]
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col7, col8 = st.columns(2)

    with col7:

        facing = st.selectbox(
            "🧭 Facing",
            [
                "North",
                "North - East",
                "North - West",
                "South",
                "South - East",
                "South -West",
                "West"
            ]
        )

    with col8:

        overlooking = st.selectbox(
            "🌳 Overlooking",
            [
                "Pool",
                "Main_Road",
                "Garden_Park"
            ]
        )

    st.markdown("<br>", unsafe_allow_html=True)

    predict = st.form_submit_button(
        "✨ Predict House Price"
    )


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if predict:

    # Start with EVERY feature expected by the PKL
    data = {
        feature: 0
        for feature in FEATURES
    }

    # Numeric features
    data["Index"] = 0
    data["Carpet Area"] = carpet_area
    data["Bathroom"] = bathroom
    data["Balcony"] = balcony
    data["Floor_Number"] = floor_number
    data["Total_Floors"] = total_floors

    # -----------------------------------------------------
    # LOCATION
    # -----------------------------------------------------

    location_column = f"newlocation_{location}"

    if location_column in data:
        data[location_column] = 1

    # -----------------------------------------------------
    # FURNISHING
    # -----------------------------------------------------

    furnishing_column = f"Furnishing_{furnishing}"

    if furnishing_column in data:
        data[furnishing_column] = 1

    # -----------------------------------------------------
    # TRANSACTION
    # -----------------------------------------------------

    transaction_map = {
        "Other": "Transaction_Other",
        "Rent/Lease": "Transaction_Rent/Lease",
        "Resale": "Transaction_Resale"
    }

    transaction_column = transaction_map[transaction]

    if transaction_column in data:
        data[transaction_column] = 1

    # -----------------------------------------------------
    # OWNERSHIP
    # -----------------------------------------------------

    ownership_column = f"Ownership_{ownership}"

    if ownership_column in data:
        data[ownership_column] = 1

    # -----------------------------------------------------
    # FACING
    # -----------------------------------------------------

    facing_column = f"facing_{facing}"

    if facing_column in data:
        data[facing_column] = 1

    # -----------------------------------------------------
    # OVERLOOKING
    # -----------------------------------------------------

    overlooking_column = f"overlooking_{overlooking}"

    if overlooking_column in data:
        data[overlooking_column] = 1

    # -----------------------------------------------------
    # DATAFRAME
    # -----------------------------------------------------

    input_data = pd.DataFrame(
        [data],
        columns=FEATURES
    )

    try:

        prediction = model.predict(input_data)[0]

        st.markdown(
            f"""
            <div class="result-box">

            <h2>🏡 Estimated House Price</h2>

            <h1>₹ {prediction:,.2f}</h1>

            <p>
            Based on the property details you entered
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:

        st.error("❌ Prediction Error")

        st.code(str(e))


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="footer">
    🏡 DreamHome • Machine Learning House Price Predictor
    </div>
    """,
    unsafe_allow_html=True
)
