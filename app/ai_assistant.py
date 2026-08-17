import os 
from groq import Groq
from dotenv import load_dotenv 
from pathlib import Path

# Find .env in project root
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

# Load environment variables from project root
load_dotenv(ENV_FILE)

# Remove invalid SSL certificate environment variable
# so httpx can use the normal certificate configuration.
os.environ.pop("SSL_CERT_FILE", None)

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def generate_churn_explanation(
    gender,
    senior,
    partner,
    dependents,
    tenure,
    phone_service,
    internet_service,
    contract,
    monthly_charges,
    prediction,
    probability,
    risk
):

    churn_status = ("likely to churn" if prediction == 1 else "likely to stay")

    prompt = f"""
You are a customer churn analyst for a telecom company.

Analyze the following customer and machine learning prediction.

Customer information:
- Gender: {gender}
- Senior Citizen: {senior}
- Partner: {partner}
- Dependents: {dependents}
- Tenure: {tenure}
- Phone Service: {phone_service}
- Internet Service: {internet_service}
- Contract: {contract}
- Monthly Charges: ${monthly_charges:.2f}
- Total Charges: ${monthly_charges * tenure:.2f}

Machine Learning Prediction:
- Prediction: {churn_status}
- Probability: {probability:.2%}
- Risk Level: {risk}

Provide a concise business-oriented explanation.

Structure your response EXACTLY as follows:

## Why?
Explain the main factors that may be associated with this customer's churn risk.

## Risk Summary
Give a short interpretation of the predicted risk.

## Recommended Actions
Suggest 2 or 3 practical customer-retention actions.

Important:
- Keep the entire response concise.
- Do not add any extra sections.
- Do not add an introduction or conclusion.
- Make sure all 3 recommended actions are completed.
- Do not stop before completing the third action.
- Do not claim that any individual feature definitely caused the prediction. 
- Explain that the prediction is based on the machine learning model.
"""

    try:

        response = client.chat.completions.create(
            model = os.getenv("GROQ_MODEL"),
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful telecom customer "  
                        "retention analyst."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature = 0.3,
            max_completion_tokens = 500
        )

        return response.choices[0].message.content

    except Exception as e:
        print("AI ERROR:", e)
        return None