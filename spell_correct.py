from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
open_ai_key = os.getenv("OPEN_AI_KEY")

client = OpenAI(api_key=open_ai_key)


import pandas as pd
import argparse
import concurrent.futures


def correct_spelling_mistakes(df):
    corrected_texts = []

    def correct_text(text, index):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant designed to correct spelling mistakes. Directly return the corrected text.",
                },
                {
                    "role": "user",
                    "content": f"Correct the spelling mistakes in the following text: {text}\nCorrected Text:",
                },
            ],
        )
        corrected_text = response.choices[0].message.content.strip()
        print(f"Corrected text {index + 1}: {corrected_text}")
        return corrected_text

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        corrected_texts = list(
            executor.map(correct_text, df["EssayText"], range(len(df["EssayText"])))
        )

    df["CorrectedSpellingEssayText"] = corrected_texts

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Correct spelling mistakes in a CSV file."
    )
    parser.add_argument(
        "input_file", type=str, help="Input CSV file containing essays."
    )
    parser.add_argument(
        "output_file", type=str, help="Output CSV file to save corrected essays."
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    df = correct_spelling_mistakes(df)
    df.to_csv(args.output_file, index=False)
