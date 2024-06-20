from typing import List

import bench.spider.schema
from bench.spider.dialogue import SpiderDatum


def serialize_schema(db_schema: bench.spider.schema.DbSchema):
    table_strs = []
    for table in db_schema.tables:
        column_strs = []
        for column in table.columns:
            column_strs.append(f'* {column.name} ({column.tpe.value}): {column.nl_name}')
        table_str = '\n'.join([table.name] + column_strs)
        table_strs.append(table_str)

    return '\n\n'.join(table_strs)


class SpiderPromptFormatter:
    def __init__(self, spider_train_data: List[SpiderDatum], db_map):
        self.spider_train_data = spider_train_data
        self.db_map = db_map

        self.llama2_chat_prompt_template = (
            """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_message_1} [/INST] {model_answer_1} </s>"""
            + '<s>[INST] {user_message_2} [/INST] {model_answer_2} </s>'
            + '<s>[INST] {user_message_3} [/INST] {model_answer_3} </s>'
            + '<s>[INST] {user_message} [/INST]'
        )

        self.system_prompt = (
            'You are a coding assistant helping an analyst answer questions over business data in SQL. '
            'More specifically, the analyst provides you a database schema '
            '(tables in the database along with their column names and types) '
            'and asks a question about the data that can be solved by issuing a SQL query to the database. '
            'In response, you write the SQL statement that answers the question. '
            'You do not provide any commentary or explanation of what the code does, '
            'just the SQL statement ending in a semicolon.'
        )

        self.user_message_template = """Here is a database schema:

{schema_str}

Please write me a SQL statement that answers the following question: {utterance}

Remember, DO NOT provide any commentary or explanation of what the code does, just the SQL statement ending in a semicolon."""

    def format_llama2(self, datum):
        spider_train_data = self.spider_train_data
        db_map = self.db_map

        llama2_chat_prompt_template = self.llama2_chat_prompt_template
        system_prompt = self.system_prompt
        user_message_template = self.user_message_template

        prompt_var_dict = {
            'system_prompt': system_prompt,
        }

        # in-context examples from training data
        for i, example_id in enumerate([10, 100, 1000], 1):
            train_datum = spider_train_data[example_id]
            user_message = user_message_template.format(
                schema_str=serialize_schema(db_map[train_datum.schema_name]),
                utterance=train_datum.utterance,
            )
            prompt_var_dict[f'user_message_{i}'] = user_message
            prompt_var_dict[f'model_answer_{i}'] = train_datum.query + ';'

        # the actual question
        user_message = user_message_template.format(
            schema_str=serialize_schema(db_map[datum.schema_name]),
            utterance=datum.utterance,
        )
        prompt_var_dict['user_message'] = user_message

        return llama2_chat_prompt_template.format(**prompt_var_dict)

    def format_openai(self, datum):
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        for example_id in [10, 100, 1000]:
            train_datum = self.spider_train_data[example_id]
            user_message = self.user_message_template.format(
                schema_str=serialize_schema(self.db_map[train_datum.schema_name]),
                utterance=train_datum.utterance,
            )
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "system", "content": train_datum.query + ";"})

        # the actual question
        user_message = self.user_message_template.format(
            schema_str=serialize_schema(self.db_map[datum.schema_name]),
            utterance=datum.utterance,
        )
        messages.append({"role": "user", "content": user_message})
        return messages
