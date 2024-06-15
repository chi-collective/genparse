#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate spider on llama2-chat models without any grammar restriction."""
import argparse
import logging
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer

import bench
from bench.spider.schema import load_schemas
from bench.spider.dialogue import load_spider_data

logger = logging.getLogger(__name__)


def serialize_schema(db_schema: bench.spider.schema.DbSchema):
    table_strs = []
    for table in db_schema.tables:
        column_strs = []
        for column in table.columns:
            column_strs.append(
                f"* {column.name} ({column.tpe.value}): {column.nl_name}"
            )
        table_str = "\n".join([table.name] + column_strs)
        table_strs.append(table_str)

    return "\n\n".join(table_strs)


class PromptFormatter:
    def __init__(self, spider_train_data, db_map):
        self.spider_train_data = spider_train_data
        self.db_map = db_map

        self.llama2_chat_prompt_template = (
            """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_message_1} [/INST] {model_answer_1} </s>"""
            + "<s>[INST] {user_message_2} [/INST] {model_answer_2} </s>"
            + "<s>[INST] {user_message_3} [/INST] {model_answer_3} </s>"
            + "<s>[INST] {user_message} [/INST]"
        )

        self.system_prompt = (
            "You are a coding assistant helping an analyst answer questions over business data in SQL. "
            "More specifically, the analyst provides you a database schema "
            "(tables in the database along with their column names and types) "
            "and asks a question about the data that can be solved by issuing a SQL query to the database. "
            "In response, you write the SQL statement that answers the question. "
            "You do not provide any commentary or explanation of what the code does, "
            "just the SQL statement ending in a semicolon."
        )

        self.user_message_template = """Here is a database schema:

{schema_str}

Please write me a SQL statement that answers the following question: {utterance}

Remember, DO NOT provide any commentary or explanation of what the code does, just the SQL statement ending in a semicolon."""

    def format(self, datum):
        spider_train_data = self.spider_train_data
        db_map = self.db_map

        llama2_chat_prompt_template = self.llama2_chat_prompt_template
        system_prompt = self.system_prompt
        user_message_template = self.user_message_template

        prompt_var_dict = {
            "system_prompt": system_prompt,
        }

        # in-context examples from training data
        for i, example_id in enumerate([10, 100, 1000], 1):
            train_datum = spider_train_data[example_id]
            user_message = user_message_template.format(
                schema_str=serialize_schema(db_map[train_datum.schema_name]),
                utterance=train_datum.utterance,
            )
            prompt_var_dict[f"user_message_{i}"] = user_message
            prompt_var_dict[f"model_answer_{i}"] = train_datum.query + ";"

        # the actual question
        user_message = user_message_template.format(
            schema_str=serialize_schema(db_map[datum.schema_name]),
            utterance=datum.utterance,
        )
        prompt_var_dict["user_message"] = user_message

        return llama2_chat_prompt_template.format(**prompt_var_dict)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-query", type=int, default=100)
    parser.add_argument("--model-size", type=str, default="7b", choices=["7b", "13b"])
    parser.add_argument("--exp-name", type=str, default="7b-100")

    return parser


def main():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
    )
    parser = get_parser()
    args = parser.parse_args()
    torch.manual_seed(0)

    raw_spider_dir = Path("bench/spider/data/spider")
    spider_schemas = load_schemas(
        schemas_path=raw_spider_dir / "tables.json", db_path=raw_spider_dir / "database"
    )

    spider_dev_data = load_spider_data(raw_spider_dir / "dev.json")
    spider_train_data = load_spider_data(raw_spider_dir / "train_spider.json")
    prompt_formatter = PromptFormatter(spider_train_data, spider_schemas)

    model = "meta-llama/Llama-2-7b-chat-hf"
    model.replace("7b", args.model_size)
    access_token = "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"

    tokenizer = AutoTokenizer.from_pretrained(model, token=access_token)
    pipe = pipeline(
        "text-generation",
        model=model,
        model_kwargs={"load_in_8bit": True},
        torch_dtype=torch.float16,
        device_map="auto",
        token=access_token,
    )

    n_query = args.n_query

    gold = []
    predicted = []

    for i, dev_datum in tqdm(enumerate(spider_dev_data[:n_query]), total=n_query):
        prompt = prompt_formatter.format(dev_datum)
        if i == 0:
            print("=" * 30 + " Example prompt " + "=" * 30)
            print(prompt)
            print("=" * 30 + "  End of prompt " + "=" * 30)
        output = pipe(prompt, do_sample=False, top_p=None, temperature=None)
        gold.append(dev_datum)
        predicted.append(output[0]["generated_text"][len(prompt) :])

    gold = spider_dev_data[:n_query]
    with open(f"spider-eval/gold-{args.exp_name}.txt", "w+") as f:
        for datum in gold:
            print(f"{datum.query}\t{datum.schema_name}", file=f)

    with open(f"spider-eval/predicted-{args.exp_name}.txt", "w+") as f:
        for datum in predicted:
            datum = datum.replace("\n", " ")
            assert "\t" not in datum
            print(datum.strip(), file=f)


if __name__ == "__main__":
    main()
