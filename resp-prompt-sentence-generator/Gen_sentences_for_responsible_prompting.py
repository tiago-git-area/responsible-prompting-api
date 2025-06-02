import os
import random
import sys
import yaml
from tqdm import tqdm
import argparse
import pandas as pd
from string import Template
from openai import OpenAI
from dotenv import dotenv_values
## --- Local ---
from utils import tools
from utils.generator_api import LLM
from utils.tools import get_column
from utils.response_quality_filters import RQF 
from utils import auxiliary_functions_for_data_pipeline as aux

# find the map of huggingface model name
def get_rits_model_list():
    url = "https://rits.fmaas.res.ibm.com/ritsapi/inferenceinfo"
    response = requests.get(url, headers={"RITS_API_KEY": config["RITS_API_KEY"]})
    if response.status_code == 200:
        return {m["model_name"]: m["endpoint"] for m in response.json()}
    else:
        raise Exception(f"Failed getting RITS model list:\n\n{response.text}")


def rits_call(url, model_id, config, prompt, temperature, max_tokens):
    try:
        
        client = OpenAI(
            api_key=config["RITS_API_KEY"],
            base_url=f'{url}/v1',
            default_headers={'RITS_API_KEY': config["RITS_API_KEY"]}
        )

        completions = client.completions.create(
            model=model_id,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return completions.to_dict()

    except Exception as exception:
        return dict(error=exception)

def load_instruction_file(instruction_path = "./seeds_and_instructions/sentence_generator_instruction.txt"):
    f = open(instruction_path)
    text = f.read()
    return text

def adapting_text(text, number_of_sentences, label):
    text = Template(text)
    return text.substitute(number_of_sentences = number_of_sentences, label = label)

if __name__ == "__main__":
    print(os.getcwd())
    parser = argparse.ArgumentParser(
        description="Script to generate sentences for responsible prompting"
    )
    parser.add_argument("--generate_model_id", required=True, type=str)
    parser.add_argument("--number_of_sentences", required=True, type=str)
    parser.add_argument("--label", required=True, type=str)
    parser.add_argument("--platform_generator", required=True, type=str)
    args = parser.parse_args()
    generate_model_id = args.generate_model_id
    number_of_sentences = int(args.number_of_sentences)
    label = args.label
    platform_generator = args.platform_generator

    parameters_base = None
    with open('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    ## find correct parameter dict based on name of the block
    for param_lst in config["blocks"]:
        if param_lst["name"] == "resp_prompt_sentence_gen":
            parameters_base = param_lst

    instruction_prompt = load_instruction_file()
    instruction_prompt = adapting_text(instruction_prompt, number_of_sentences, label)
    print(instruction_prompt)

    _model_url="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mixtral-8x7b-instruct-v01"
    _model_id="mistralai/mixtral-8x7B-instruct-v0.1"

    config = dotenv_values(".env")
    response = rits_call(_model_url, _model_id, config, instruction_prompt, 0.0, 1500)
    print(response["choices"][0]["text"])