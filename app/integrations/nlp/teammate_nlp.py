import os
import re
import joblib


# ============================================================
# 1. MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "jharkhand_multilingual_model"
)

CLASSIFIER_PATH = os.path.join(
    MODEL_DIR,
    "classifier.joblib"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.joblib"
)

LABEL_ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "label_encoder.joblib"
)


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

print("Loading Jharkhand NLP model...")

classifier = joblib.load(CLASSIFIER_PATH)

vectorizer = joblib.load(
    VECTORIZER_PATH
)

label_encoder = joblib.load(
    LABEL_ENCODER_PATH
)

print("NLP model loaded successfully.")


# ============================================================
# 3. CATEGORY INFORMATION
# ============================================================

CATEGORY_INFO = {

    "Accessibility": {
        "problem_type": "Accessibility Issue",
        "department": "Social Welfare / Municipal Department",
        "domain": "Accessibility",
        "suggested_action":
            "Inspect and improve accessibility facilities"
    },

    "Agriculture": {
        "problem_type": "Agricultural Issue",
        "department": "Agriculture Department",
        "domain": "Agriculture",
        "suggested_action":
            "Inspect the agricultural issue and provide required support"
    },

    "Disaster Management": {
        "problem_type": "Disaster-related Issue",
        "department": "Disaster Management Department",
        "domain": "Disaster Management",
        "suggested_action":
            "Assess the situation and provide emergency assistance"
    },

    "Education": {
        "problem_type": "Education-related Issue",
        "department": "Education Department",
        "domain": "Education",
        "suggested_action":
            "Inspect the educational facility and take corrective action"
    },

    "Environment": {
        "problem_type": "Environmental Issue",
        "department": "Environment Department",
        "domain": "Environment",
        "suggested_action":
            "Inspect the environmental problem and take corrective action"
    },

    "Garbage & Waste Management": {
        "problem_type": "Uncollected Waste",
        "department": "Sanitation Department",
        "domain": "Waste Management",
        "suggested_action":
            "Arrange immediate waste collection"
    },

    "Healthcare": {
        "problem_type": "Healthcare Service Issue",
        "department": "Health Department",
        "domain": "Healthcare",
        "suggested_action":
            "Inspect the healthcare facility and provide required services"
    },

    "Other Societal Issues": {
        "problem_type": "Societal Issue",
        "department": "Relevant Government Department",
        "domain": "Societal Issues",
        "suggested_action":
            "Investigate the reported societal issue"
    },

    "Public Administration": {
        "problem_type": "Administrative Issue",
        "department": "District Administration",
        "domain": "Public Administration",
        "suggested_action":
            "Review the complaint and take administrative action"
    },

    "Public Services": {
        "problem_type": "Public Service Issue",
        "department": "Relevant Public Service Department",
        "domain": "Public Services",
        "suggested_action":
            "Inspect the service issue and restore the required service"
    },

    "Road & Urban Infrastructure": {
        "problem_type": "Road Damage / Potholes",
        "department": "Road / Municipal Department",
        "domain": "Urban Infrastructure",
        "suggested_action":
            "Inspect and repair damaged roads or potholes"
    },

    "Rural Livelihoods": {
        "problem_type": "Rural Livelihood Issue",
        "department": "Rural Development Department",
        "domain": "Rural Livelihoods",
        "suggested_action":
            "Investigate the issue and provide livelihood support"
    },

    "Street Light & Energy": {
        "problem_type": "Non-functional Street Light",
        "department": "Municipal / Electrical Department",
        "domain": "Urban Infrastructure",
        "suggested_action":
            "Inspect and repair non-functional street lights"
    },

    "Urban Development": {
        "problem_type": "Urban Development Issue",
        "department": "Urban Development Department",
        "domain": "Urban Development",
        "suggested_action":
            "Inspect the infrastructure issue and take corrective action"
    },

    "Water Management": {
        "problem_type": "Water Supply Issue",
        "department": "Water Supply Department",
        "domain": "Water Management",
        "suggested_action":
            "Inspect and restore the water supply"
    }
}


# ============================================================
# 4. LANGUAGE DETECTION
# ============================================================

def detect_language(text):

    devanagari_count = len(
        re.findall(
            r"[\u0900-\u097F]",
            text
        )
    )

    english_count = len(
        re.findall(
            r"[A-Za-z]",
            text
        )
    )

    words = text.lower().split()

    hinglish_words = {
        "hamare",
        "hamara",
        "hamari",
        "area",
        "mein",
        "hai",
        "hain",
        "bahut",
        "problem",
        "road",
        "paani",
        "pani",
        "garbage",
        "light",
        "nahi",
        "nahin",
        "ho",
        "raha",
        "rhi",
        "liye",
        "wala",
        "wali",
        "ke",
        "ko",
        "se"
    }

    hinglish_count = sum(
        1
        for word in words
        if word.strip(".,!?") in hinglish_words
    )

    if devanagari_count > 0 and english_count == 0:
        return "Hindi"

    if hinglish_count >= 2:
        return "Hinglish"

    return "English"


# ============================================================
# 5. DURATION EXTRACTION
# ============================================================

def extract_duration(text):

    # ----------------------------------------
    # Hindi number words
    # ----------------------------------------

    hindi_numbers = {

        "एक": "1",
        "दो": "2",
        "तीन": "3",
        "चार": "4",
        "पांच": "5",
        "पाँच": "5",
        "छह": "6",
        "छः": "6",
        "सात": "7",
        "आठ": "8",
        "नौ": "9",
        "दस": "10"
    }

    # ----------------------------------------
    # Hindi number + time unit
    # ----------------------------------------

    for word, number in hindi_numbers.items():

        pattern = (
            rf"{word}\s*"
            r"(दिन|हफ्ता|हफ्ते|हफ्तों|"
            r"महीना|महीने|महीनों|"
            r"घंटा|घंटे|घंटों)"
        )

        match = re.search(
            pattern,
            text
        )

        if match:

            unit = match.group(1)

            if "दिन" in unit:
                return f"{number} days"

            if "हफ्त" in unit:
                return f"{number} weeks"

            if "मही" in unit:
                return f"{number} months"

            if "घंट" in unit:
                return f"{number} hours"

    # ----------------------------------------
    # Numeric duration
    # ----------------------------------------

    patterns = [

        r"\b\d+\s*(?:day|days|din)\b",

        r"\b\d+\s*(?:week|weeks|hafte|hafton)\b",

        r"\b\d+\s*(?:month|months|mahine|mahino)\b",

        r"\b\d+\s*(?:hour|hours|ghante)\b"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(0)

    return "Not specified"


# ============================================================
# 6. AFFECTED PEOPLE
# ============================================================

def extract_affected_people(text):

    lower = text.lower()

    if any(
        word in lower
        for word in [
            "students",
            "student",
            "bachche",
            "children",
            "school"
        ]
    ):
        return "Students"

    if any(
        word in lower
        for word in [
            "residents",
            "resident",
            "people",
            "public",
            "log"
        ]
    ):
        return "Residents"

    if any(
        word in lower
        for word in [
            "farmers",
            "farmer",
            "kisan"
        ]
    ):
        return "Farmers"

    if any(
        word in lower
        for word in [
            "villagers",
            "village",
            "gaon",
            "गांव",
            "गाँव"
        ]
    ):
        return "Villagers"

    return "Not specified"


# ============================================================
# 7. LOCATION EXTRACTION
# ============================================================

def extract_location(text):

    # Hindi locations
    hindi_patterns = [

        r"(मोहल्ले|मोहल्ला)",

        r"(गांव|गाँव)",

        r"(कॉलोनी)",

        r"(शहर)",

        r"(सड़क)",

        r"(क्षेत्र)"
    ]

    for pattern in hindi_patterns:

        match = re.search(
            pattern,
            text
        )

        if match:
            return match.group(1)

    # English / Hinglish
    patterns = [

        r"\b(?:in|near|at|from)\s+"
        r"(?:our\s+)?"
        r"([A-Za-z][A-Za-z\s]{2,30})"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            location = match.group(1).strip()

            # Avoid capturing unnecessary words
            location = location.split(
                " and "
            )[0]

            return location

    return "Not specified"


# ============================================================
# 8. SEVERITY
# ============================================================

def determine_severity(
    text,
    category
):

    lower = text.lower()

    high_words = [

        "dangerous",
        "unsafe",
        "emergency",
        "critical",
        "accident",
        "flood",
        "death",
        "very unsafe",

        "बहुत खतरनाक",
        "खतरा",
        "आपातकाल",
        "बहुत अंधेरा",
        "अंधेरा रहता है",
        "रात में अंधेरा",
        "जान का खतरा"
    ]

    if any(
        word in lower
        for word in high_words
    ):
        return "High"

    return "Medium"


# ============================================================
# 9. PRIORITY
# ============================================================

def determine_priority(
    severity,
    duration,
    text
):

    lower = text.lower()

    if severity == "High":
        return "High"

    if any(
        word in lower
        for word in [
            "urgent",
            "immediately",
            "emergency",
            "jaldi",
            "turant",
            "तुरंत",
            "तत्काल",
            "आपातकाल"
        ]
    ):
        return "High"

    if duration != "Not specified":
        return "Medium"

    return "Medium"


# ============================================================
# 10. URGENCY
# ============================================================

def determine_urgency(
    priority
):

    if priority == "High":
        return "High"

    return "Medium"


# ============================================================
# 11. MAIN PREDICTION FUNCTION
# ============================================================

def predict_complaint(
    complaint
):

    if not complaint:

        raise ValueError(
            "Complaint text cannot be empty."
        )

    complaint = complaint.strip()

    if not complaint:

        raise ValueError(
            "Complaint text cannot be empty."
        )

    # ----------------------------------------
    # Language
    # ----------------------------------------

    language = detect_language(
        complaint
    )

    # ----------------------------------------
    # TF-IDF transformation
    # ----------------------------------------

    features = vectorizer.transform(
        [complaint]
    )

    # ----------------------------------------
    # Category prediction
    # ----------------------------------------

    prediction = classifier.predict(
        features
    )

    category = label_encoder.inverse_transform(
        prediction
    )[0]

    # ----------------------------------------
    # Category information
    # ----------------------------------------

    info = CATEGORY_INFO.get(
        category,
        {
            "problem_type":
                "Not specified",

            "department":
                "Relevant Government Department",

            "domain":
                "Societal Issues",

            "suggested_action":
                "Investigate the reported issue"
        }
    )

    # ----------------------------------------
    # Information extraction
    # ----------------------------------------

    duration = extract_duration(
        complaint
    )

    affected_people = extract_affected_people(
        complaint
    )

    location = extract_location(
        complaint
    )

    severity = determine_severity(
        complaint,
        category
    )

    priority = determine_priority(
        severity,
        duration,
        complaint
    )

    urgency = determine_urgency(
        priority
    )

    # ----------------------------------------
    # Final result
    # ----------------------------------------

    result = {

        "complaint":
            complaint,

        "language":
            language,

        "primary_category":
            category,

        "secondary_category":
            "Not specified",

        "problem_type":
            info["problem_type"],

        "severity":
            severity,

        "priority":
            priority,

        "location":
            location,

        "landmark":
            "Not specified",

        "duration":
            duration,

        "cause":
            "Not specified",

        "affected_people":
            affected_people,

        "urgency":
            urgency,

        "department":
            info["department"],

        "suggested_action":
            info["suggested_action"],

        "domain":
            info["domain"]
    }

    return result


# ============================================================
# 12. COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 50)
    print(
        "JHARKHAND AI SOCIETAL PROBLEM DETECTOR"
    )
    print("=" * 50)

    print()
    print(
        "Supported languages:"
    )

    print(
        "Hindi | English | Hinglish"
    )

    print()

    while True:

        complaint = input(
            "Enter complaint "
            "(type 'exit' to stop): "
        )

        if complaint.lower().strip() == "exit":

            print()
            print(
                "Exiting program..."
            )

            break

        try:

            result = predict_complaint(
                complaint
            )

            print()
            print("=" * 50)

            print(
                "AI COMPLAINT ANALYSIS"
            )

            print("=" * 50)

            print()

            for key, value in result.items():

                print(
                    f"{key}: {value}"
                )

            print()

            print("=" * 50)

            print()

        except Exception as e:

            print()
            print(
                "Error:",
                str(e)
            )

            print()