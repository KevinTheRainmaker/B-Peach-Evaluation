import requests
import json
import re
import random
import csv
import argparse
from tqdm import tqdm
from konlpy.tag import Okt
import os

API_KEY = os.getenv('API_KEY') 

# API_KEY = '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'
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

def get_response_from_model(model_id, prompt, test, example=None):
    headers = {
        'Authorization': f'Bearer {API_KEY}',
    }

    data = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": prompt,
            }
        ],
    }

    if example:
        if isinstance(example, list):
            for ex in example:
                data["messages"].append({
                    "role": "user",
                    "content": ex['user'],
                })
                data["messages"].append({
                    "role": "assistant",
                    "content": ex['assistant'],
                })

    data["messages"].append({
        "role": "user",
        "content": test,
    })

    try:
        response = requests.post(OPENROUTER_ENDPOINT, headers=headers, data=json.dumps(data))
        response.raise_for_status()

        output = response.json()['choices'][0]['message']['content']


        # TODO: Pydantic Parser?

        return output

    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")

def extract_annotated_spans(text: str):
    pattern = r"<span\s+adaptation='no'>(.*?)<\/span>"
    matches = re.findall(pattern, text, re.DOTALL)
    unique_matches = list(set(matches))  # Ensure unique matches only
    return len(unique_matches), unique_matches

def create_test_data(json_data):
    test_data = []
    okt = Okt()

    for passage in json_data:
        if passage.startswith('"') and passage.endswith('"'):
            passage = passage[1:-1]  # Remove quotation marks
        inputs = [passage]  # Treat the entire passage as one input
        for input in inputs:
            nouns = [word for word, tag in okt.pos(input) if tag == 'Noun'] # Select only Noun
            if len(nouns) > 1:
                num_tags = random.randint(1, min(5, len(nouns)))  # Random number of tags (1 to 5 or noun count)
                selected_words = random.sample(nouns, num_tags)
                tagged_sentence = input
                for word in selected_words:
                    tagged_sentence = re.sub(
                        rf"(?<!<span adaptation='no'>){word}(?!<\/span>)",
                        f"<span adaptation='no'>{word}</span>",
                        tagged_sentence,
                        count=1
                    )
                test_data.append(tagged_sentence)
    return test_data


def main(args):
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        prompt = f.read()

    with open(args.example_file, 'r', encoding='utf-8') as f:
        examples = json.load(f)

    with open(args.input_json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    all_results = []

    for iteration in range(args.iterations):
        print(f"Iteration {iteration + 1}:")

        test_data = create_test_data(json_data)

        for test in tqdm(test_data, desc="Evaluating..."):
            try:
                response = get_response_from_model(args.model_id, prompt, test, examples)
                
                pattern = r"3\. 정리.*?\n([\s\S]+)$"
                extracted = re.split(pattern, response)[1] if "3. 정리" in response else ""

                l_o, ori_word = extract_annotated_spans(test)
                l_r, res_word = extract_annotated_spans(extracted)

                matches = set(ori_word).intersection(set(res_word))
                em = len(matches) / l_o if l_o else 0

                all_results.append([test, ori_word, extracted, em])

            except Exception as e:
                print(f"Error processing test: {test}\n{e}")

    with open(args.output_csv_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["original_passage", "tagged_words", "response_summary", "em_score"])
        writer.writerows(all_results)
    
    print(f"Results saved to {args.output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json_path", type=str, default="input.json", help="Path to the input JSON file.")
    parser.add_argument("--prompt_file", type=str, help="Path to the prompt file", default="prompt.txt")
    parser.add_argument("--example_file", type=str, help="Path to the example file (.json)", default="examples.json")

    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations to run.")
    
    parser.add_argument("--output_csv_path", type=str, required=True, help="Path to save the output CSV file.")
    parser.add_argument("--model_id", type=str, help="Model ID of the model you want to use", default="anthropic/claude-3.5-sonnet")

    args = parser.parse_args()
    main(args)