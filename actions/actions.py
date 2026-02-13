import pandas as pd
import re
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load datasets
disease_symptoms = pd.read_csv(os.path.join(BASE_DIR, "disease_symptoms.csv"))
disease_info = pd.read_csv(os.path.join(BASE_DIR, "disease_info.csv"))

# Make all symptom names lowercase
all_symptoms = [col.lower() for col in disease_symptoms.columns if col != "prognosis"]

def clean_symptom_input(user_input: str) -> List[str]:
    """
    Extract valid symptoms from user input by removing filler words.
    """
    text = user_input.lower()
    filler_words = ["i have", "i am", "i feel", "i'm", "having",
                    "suffering from", "got", "with", "also", "and"]
    for word in filler_words:
        text = text.replace(word, " ")

    tokens = re.split(r"[,\s]+", text)
    return [t.strip() for t in tokens if t.strip() in all_symptoms]

class ActionPredictDisease(Action):
    def name(self) -> Text:
        return "action_predict_disease"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Get user's reported symptoms
        user_message = tracker.latest_message.get("text", "")
        new_symptoms = clean_symptom_input(user_message)

        if not new_symptoms:
            dispatcher.utter_message(text="I couldn’t recognize any valid symptoms. Could you rephrase?")
            return []

        # Merge with previously stored symptoms
        stored_symptoms = tracker.get_slot("symptoms") or []
        updated_symptoms = list(set(stored_symptoms + new_symptoms))

        # Filter candidate diseases based on known symptoms
        candidate_diseases = disease_symptoms.copy()
        for sym in updated_symptoms:
            if sym in candidate_diseases.columns:
                candidate_diseases = candidate_diseases[candidate_diseases[sym] == 1]

        possible_diseases = candidate_diseases["prognosis"].tolist()

        if not possible_diseases:
            dispatcher.utter_message(text="Sorry, I could not match your symptoms to any known disease.")
            return [SlotSet("symptoms", updated_symptoms), SlotSet("possible_diseases", [])]

        # If exactly one disease remains, give full info
        if len(possible_diseases) == 1:
            final_disease = possible_diseases[0]
            details = disease_info[disease_info["disease"] == final_disease]
            if not details.empty:
                description = details["description"].values[0]
                treatment = details["treatment"].values[0]
                prevention = details["prevention"].values[0]
                msg = (f"Based on your symptoms, the most likely disease is **{final_disease}**.\n\n"
                       f"📋 Description: {description}\n"
                       f"💊 Treatment: {treatment}\n"
                       f"🛡 Prevention: {prevention}")
            else:
                msg = f"Based on your symptoms, the most likely disease is **{final_disease}**."

            dispatcher.utter_message(text=msg)
            return [
                SlotSet("symptoms", updated_symptoms),
                SlotSet("possible_diseases", [final_disease]),
                SlotSet("predicted_disease", final_disease)
            ]

        # If multiple diseases remain, ask for additional symptoms
        remaining_symptoms = []
        for sym in all_symptoms:
            if sym not in updated_symptoms and sym in candidate_diseases.columns:
                if candidate_diseases[sym].sum() > 0:
                    remaining_symptoms.append(sym)

        if remaining_symptoms:
            dispatcher.utter_message(
                text=f"I see multiple possible diseases. Do you also have any of these symptoms? {', '.join(remaining_symptoms[:5])}"
            )
        else:
            dispatcher.utter_message(
                text=f"Based on your symptoms, possible diseases are: {', '.join(possible_diseases)}."
            )

        return [
            SlotSet("symptoms", updated_symptoms),
            SlotSet("possible_diseases", possible_diseases),
            SlotSet("remaining_symptoms_to_ask", remaining_symptoms[:5])
        ]

class ActionGetDiseaseInfo(Action):
    def name(self) -> Text:
        return "action_get_disease_info"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        disease = tracker.get_slot("predicted_disease")
        if not disease:
            dispatcher.utter_message(text="I don’t know which disease to explain. Could you tell me?")
            return []

        info = disease_info[disease_info["disease"] == disease]
        if not info.empty:
            description = info["description"].values[0]
            treatment = info["treatment"].values[0]
            prevention = info["prevention"].values[0]
            dispatcher.utter_message(
                text=f"📋 Description: {description}\n💊 Treatment: {treatment}\n🛡 Prevention: {prevention}"
            )
        else:
            dispatcher.utter_message(text=f"Sorry, I don’t have details about {disease}.")

        return []
