import requests
import json
import argparse
import re
from tqdm import tqdm
import os

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
        "model": model_id,  # Model ID for the specific model you want to use
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

    return len(matches), matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser() # argparse에 미친 자..
    parser.add_argument("--model_id", type=str, help="Model ID of the model you want to use", default="anthropic/claude-3.5-sonnet")
    parser.add_argument("--prompt_file", type=str, help="Path to the prompt file", default="prompt.txt")
    parser.add_argument("--example_file", type=str, help="Path to the example file (.json)", default="examples.json")
    parser.add_argument("--test_file", type=str, help="Path to the test file that contains test examples", default="test.json")
    parser.add_argument("--output_file", type=str, help="Output filename; all results including responses and EM score.", default="results.json")
    args = parser.parse_args() 
    
    print("Reading files...")
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        prompt = f.read()
    
    with open(args.example_file, 'r', encoding='utf-8') as f:
        examples = json.load(f)
        
    with open(args.test_file, 'r', encoding='utf-8') as f:
        tests = json.load(f)
    print("Files read successfully.")
    
    results = []
    total_spans = 0
    total_matches = 0
    
    for test in tqdm(tests, desc="Evaluating..."):
        response = get_response_from_model(args.model_id, prompt, test, examples) # 모델 응답 받기
        try:
            pattern = r"3\. 정리.*?\n([\s\S]+)$"
            extracted = re.split(pattern, response)[1] # 전체 응답에서 3. 정리 부분(번안문)만 추출
            
            l_o, ori_word = extract_annotated_spans(test)
            l_r, res_word = extract_annotated_spans(extracted)
            
            matches = set(ori_word).intersection(set(res_word))
            
            em = len(matches) / l_o if l_o else 0 # EM score 계산
            
            results.append({
                "original_passage": test,
                "response": response,
                "original_spans": ori_word,
                "response_spans": res_word,
                "em_score": em
            })
            
            total_spans += l_o
            total_matches += len(matches)
            
        except Exception as e:
            print(f"Error encountered: {str(e)}")
            continue
    
    print(f"Total EM Score: {total_matches / total_spans}") # 전체 EM score 출력
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)