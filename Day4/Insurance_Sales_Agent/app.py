import http.client
import json
import google.generativeai as genai

# ----------------------------
# 1. Setup Gemini API
# ----------------------------
GEMINI_API_KEY = "AIzaSyCp8H9Ihvgujw76b56eIVQOAK8Jr92YBpo"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# ----------------------------
# 2. Fetch insurance data from RapidAPI
# ----------------------------
def fetch_insurance_data():
    conn = http.client.HTTPSConnection("health-insurance-market.p.rapidapi.com")

    headers = {
        'x-rapidapi-key': "28e735e435msh7f87315bb9bfbe7p12bf53jsna5d72e8b4c5e",
        'x-rapidapi-host': "health-insurance-market.p.rapidapi.com"
    }

    conn.request("GET", "/", headers=headers)
    res = conn.getresponse()
    data = res.read()
    return json.loads(data.decode("utf-8"))

# ----------------------------
# 3. Recommend policy using Gemini
# ----------------------------
def recommend_insurance(user_input, policies):
    context = "Here are some available insurance policies:\n\n"

    # Adjust this based on actual API response structure
    plans = policies.get("plans") or policies.get("data") or policies
    if not isinstance(plans, list):
        context += json.dumps(policies, indent=2)
    else:
        for p in plans:
            name = p.get("plan_name", p.get("name", "Unknown Plan"))
            coverage = p.get("coverage", p.get("coverage_type", "N/A"))
            benefits = p.get("benefits", p.get("features", []))
            benefits_str = ", ".join(benefits) if isinstance(benefits, list) else str(benefits)
            context += f"- {name} | Coverage: {coverage} | Benefits: {benefits_str}\n"

    prompt = f"""{context}

Based on this user's profile, recommend the best suitable insurance plan:

{user_input}

Provide a short, clear answer with plan name and reason.
"""

    response = model.generate_content(prompt)
    return response.text

# ----------------------------
# 4. CLI Interface
# ----------------------------
def main():
    print("🔍 Welcome to the AI-powered Insurance Recommender\n")

    age = input("👤 Enter your age: ")
    coverage = input("🏥 Coverage type (individual/family): ")
    dependents = input("👨‍👩‍👧 Number of dependents: ")
    special = input("💬 Any special requirements? (e.g., dental, wellness, senior): ")

    user_profile = f"""
    Age: {age}
    Coverage Type: {coverage}
    Dependents: {dependents}
    Special Requirements: {special}
    """

    print("\n📡 Fetching latest insurance plans from RapidAPI...")
    policies = fetch_insurance_data()

    print("\n🤖 Asking Gemini for the best recommendation...")
    recommendation = recommend_insurance(user_profile, policies)

    print("\n✅ Recommended Plan:\n")
    print(recommendation)

# ----------------------------
# 5. Run
# ----------------------------
if __name__ == "__main__":
    main()
