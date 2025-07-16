# chatbot.py

import google.generativeai as genai

genai.configure(api_key="AIzaSyCLTF0ghfRFpHorDR-3XK3pv_5pK-xXT2E")

model = genai.GenerativeModel("models/gemini-2.5-flash")

def ai_chatbot(user_input, username="User", health_history=None, latest_data=None):
    try:
        # Prepare health history string
        history_text = ""
        if health_history:
            history_text = "User's past health check history:\n"
            for i, row in enumerate(health_history[-5:], 1):
                history_text += (
                    f"{i}. Age: {row['age']}, BMI: {row['bmi']}, BP: {row['blood_pressure']}, "
                    f"Glucose: {row['glucose']}, Insulin: {row['insulin']}, Risk: {'High' if row['result'] else 'Low'}\n"
                )

        # Add recent values for direct facts
        recent_metrics = ""
        if latest_data:
            recent_metrics = (
                f"\nLatest Checkup:\n"
                f"Age: {latest_data['age']}, BMI: {latest_data['bmi']}, BP: {latest_data['blood_pressure']}, "
                f"Glucose: {latest_data['glucose']}, Insulin: {latest_data['insulin']}, "
                f"Risk: {'High' if latest_data['result'] else 'Low'}\n"
            )

        prompt = (
            f"You are a friendly health assistant for {username} 🧑‍⚕️.\n"
            "Answer clearly in 1–3 short lines with emojis. 😊💡\n"
            "Use the latest health check to answer fact questions like age, BMI, or glucose.\n"
            f"{recent_metrics}\n{history_text}\n"
            f"{username} says: {user_input}\n"
            "Your response:"
        )

        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini Error: {str(e)}"