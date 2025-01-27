import requests
import json
import re
import random
import csv
import argparse
from tqdm import tqdm
from konlpy.tag import Okt
import os
from datetime import datetime

if os.environ.get("GITHUB_ACTIONS") is None:
    from dotenv import load_dotenv
    print('Load API Key from Local .env file..')
    load_dotenv()
else:
    print('Load API Key from GitHub Secrets..')
    
API_KEY = os.getenv('API_KEY') 

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

    for passage in tqdm(json_data, desc="Processing passages..."):  # tqdm을 passage 단위로 적용
        if passage.startswith('"') and passage.endswith('"'):
            passage = passage[1:-1]  # Remove quotation marks

        # 텍스트를 문장 단위로 분리
        sentences = re.split(r'[.!?]', passage)
        sentences = [s.strip() for s in sentences if s.strip()]

        # 문장별 명사 태깅
        tagged_sentences = []
        all_nouns = []
        for sentence in sentences:
            nouns = [word for word, tag in okt.pos(sentence) if tag == 'Noun']
            all_nouns.extend(nouns)

        # 태깅할 명사 선택 (1~5개)
        if len(all_nouns) > 1:
            num_tags = random.randint(1, min(5, len(all_nouns)))  # 1~5개 랜덤 선택
            selected_words = random.sample(all_nouns, num_tags)

            # 선택된 단어를 passage 전체에서 태깅
            tagged_passage = passage
            for word in selected_words:
                tagged_passage = re.sub(
                    rf"(?<!<span adaptation='no'>){word}(?!<\/span>)",
                    f"<span adaptation='no'>{word}</span>",
                    tagged_passage,
                    count=1
                )
            test_data.append(tagged_passage)
        else:
            test_data.append(passage)

    print(len(test_data))
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
    now = datetime.now()
    file_name = now.strftime("output_%y%m%d_%H%M.csv")
    os.makedirs('results', exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json_path", type=str, default="data/input.json", help="Path to the input JSON file.")
    parser.add_argument("--prompt_file", type=str, help="Path to the prompt file", default="data/prompt.txt")
    parser.add_argument("--example_file", type=str, help="Path to the example file (.json)", default="data/examples.json")

    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations to run.")
    
    parser.add_argument("--output_csv_path", type=str, default=f'results/{file_name}', help="Path to save the output CSV file.")
    parser.add_argument("--model_id", type=str, help="Model ID of the model you want to use", default="anthropic/claude-3.5-sonnet")

    args = parser.parse_args()
    main(args)