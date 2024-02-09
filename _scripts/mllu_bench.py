import json
import os
from datasets import Dataset, DatasetDict
import os.path as osp
import csv
from typing import List
import argparse
from argparse import Namespace
import time

from vllm import LLM, SamplingParams

class MMLUDataset:
    mmlu_all_sets = [
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "electrical_engineering",
        "astronomy",
        "anatomy",
        "abstract_algebra",
        "machine_learning",
        "clinical_knowledge",
        "global_facts",
        "management",
        "nutrition",
        "marketing",
        "professional_accounting",
        "high_school_geography",
        "international_law",
        "moral_scenarios",
        "computer_security",
        "high_school_microeconomics",
        "professional_law",
        "medical_genetics",
        "professional_psychology",
        "jurisprudence",
        "world_religions",
        "philosophy",
        "virology",
        "high_school_chemistry",
        "public_relations",
        "high_school_macroeconomics",
        "human_sexuality",
        "elementary_mathematics",
        "high_school_physics",
        "high_school_computer_science",
        "high_school_european_history",
        "business_ethics",
        "moral_disputes",
        "high_school_statistics",
        "miscellaneous",
        "formal_logic",
        "high_school_government_and_politics",
        "prehistory",
        "security_studies",
        "high_school_biology",
        "logical_fallacies",
        "high_school_world_history",
        "professional_medicine",
        "high_school_mathematics",
        "college_medicine",
        "high_school_us_history",
        "sociology",
        "econometrics",
        "high_school_psychology",
        "human_aging",
        "us_foreign_policy",
        "conceptual_physics",
    ]
    question_sep='\n\n'
    def __init__(self, path:str, name:str):
        assert name in self.mmlu_all_sets
        self.name = name
        self.dataset = self.load(path, name)

    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()
        for split in ['dev', 'test']:
            raw_data = []
            filename = osp.join(path, split, f'{name}_{split}.csv')
            with open(filename, encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    assert len(row) == 6
                    raw_data.append({
                        'input': row[0],
                        'A': row[1],
                        'B': row[2],
                        'C': row[3],
                        'D': row[4],
                        'target': row[5],
                    })
            dataset[split] = Dataset.from_list(raw_data)
        return dataset

    @property
    def test(self):
        return self.dataset['test']
    
    @property
    def fewshot(self):
        return self.dataset['dev']

    def get_fewshot_prefix(self, nshot=5):
        assert nshot > 0
        prefix = []
        for i in range(nshot):
            example, _ = self.format_read(split='dev', idx=i, with_target=True)
            prefix.append(example)
        return self.question_sep.join(prefix) + self.question_sep
    
    def get_question(self, idx):
        return self.format_read(split='test', idx=idx, with_target=False)

    def format_read(self, split:str, idx:int, with_target=False):
        item = self.dataset[split][idx]
        _hint = f'There is a single choice question about {self.name.replace("_", " ")}. Answer the question by replying A, B, C or D.'
        prompt = f"{_hint}\nQuestion: {item['input']}\nA. {item['A']}\nB. {item['B']}\nC. {item['C']}\nD. {item['D']}\nAnswer: "
        target = item['target']
        if with_target:
            prompt += f"\n{target}"
        return prompt, target
    
    def get_num_questions(self):
        return len(self.dataset['test'])


def save_dict_to_json(dict_save:dict, json_path:str):
    with open(json_path, mode='w') as f:
        json.dump(dict_save, f, indent=4)


def calculate_accuracy(json_path:str):
    with open(json_path) as f:
        preds:dict = json.load(f)

    def first_capital_postprocess(text: str) -> str:
        for t in text:
            if t.isupper():
                return t
        return ''
    
    total_cnt = 0
    correct_cnt = 0
    for idx, results in preds.items():
        pred = results["prediction"]
        pred = first_capital_postprocess(pred)
        gold = results["gold"]
        total_cnt += 1
        correct_cnt += (pred == gold)
    accuracy = correct_cnt / total_cnt
    
    return correct_cnt, accuracy, total_cnt


def main(args:Namespace):
    template = "{__SYS_PROMPT}{__USR_PROMPT}"
    # No random sampling
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    llm = LLM(model=args.model,
              enable_relay_attention=args.enable_relay,
              dtype=args.dtype)


    _total_times=[]
    _prefix_times=[]
    _total_cnts=[]
    _correct_cnts=[]
    _accs=[]
    for subset in MMLUDataset.mmlu_all_sets:
        print(f"Processing {subset}")
        dataset = MMLUDataset(path=args.mmlu_root, name=subset)
        num_questions = dataset.get_num_questions()
        all_items = [ dataset.get_question(i) for i in range(num_questions) ]
        prompts = [ question for (question, _) in all_items ]
        golds = [ gold for (_, gold) in all_items ]
        prefix = dataset.get_fewshot_prefix(args.nshot)
        start = time.perf_counter()
        if args.enable_relay:
            llm.fill_prefix_kv_cache(prefix)
        else:
            prompts = [ template.format(__SYS_PROMPT=prefix, __USR_PROMPT=x) for x in prompts ]
        time_prefix = time.perf_counter() - start
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        time_all = time.perf_counter() - start
        # save results
        os.makedirs(args.save_dir, exist_ok=True)
        predictions = {
            idx:{'prompt': outputs[idx].prompt,
                'prediction': outputs[idx].outputs[0].text,
                'gold': golds[idx]}
            for idx in range(num_questions)
        }
        predictions_path = osp.join(args.save_dir, f'predictions_{subset}.json')
        save_dict_to_json(predictions, predictions_path)
        # calculate metrics
        correct_cnt, accuracy, total_cnt = calculate_accuracy(predictions_path)
        assert total_cnt == num_questions
        # save statistics for the current subset
        summary = dict(time_prefix=time_prefix,
                       time_all=time_all,
                       num_requests=num_questions,
                       num_correct=correct_cnt,
                       accuracy=accuracy,
                    )
        print(summary)
        summary_path = osp.join(args.save_dir, f'summary_{subset}.json')
        save_dict_to_json(summary, summary_path)
        # 
        _prefix_times.append(time_prefix)
        _total_times.append(time_all)
        _total_cnts.append(total_cnt)
        _correct_cnts.append(correct_cnt)
        _accs.append(accuracy)
        
    global_summary = dict(
        prefix_time = sum(_prefix_times),
        total_time = sum(_total_times),
        correct_cnt = sum(_correct_cnts),
        total_cnt = sum(_total_cnts),
        average_acc = sum(_accs) / len(_accs),
        global_acc = sum(_correct_cnts) / sum(_total_cnts)
    )
    print(global_summary)
    global_summary_path = osp.join(args.save_dir, f'summary_all.json')
    save_dict_to_json(global_summary, global_summary_path)


if __name__ == "__main__":
    def str2bool(v:str):
        """
        Converts string to bool type; enables command line 
        arguments in the format of '--arg1 true --arg2 false'
        """
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser(description="Run MLLU benchmark.")
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--mmlu-root', type=str, default='data/mmlu')
    parser.add_argument('--nshot', type=int, default=5)
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--enable-relay', type=str2bool, default=False)
    parser.add_argument(
            '--dtype',
            type=str,
            default="auto",
            choices=[
                'auto', 'half', 'float16', 'bfloat16', 'float', 'float32'
            ],
            help='data type for model weights and activations. '
            'The "auto" option will use FP16 precision '
            'for FP32 and FP16 models, and BF16 precision '
            'for BF16 models.')
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = f'model_{os.path.basename(args.model)}' \
            f'.nshot_{args.nshot}.relay_{args.enable_relay}.outlen_{args.max_tokens}'
    # args = Namespace(
    #     max_tokens=128,
    #     model='/mnt/cachenew2/zhulei1/huggingface/local/Llama-2-7b-hf',
    #     enable_relay=True,
    #     mllu_root='/mnt/lustrenew/zhulei1/ssd_cache/opencompass/data/mmlu',
    #     nshot=5,
    #     save_dir='outputs/mllu_bench/')

    main(args)