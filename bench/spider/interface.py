from path import Path

from bench.spider.dialogue import load_spider_data
from bench.spider.schema import load_schemas
from bench.spider.evaluator import Evaluator

import genparse
import sqlite3
import pandas as pd


class SpiderInterface:
    def __init__(self, root=None):
        if root is None:
            root = Path(genparse.__file__).dirname() / '..' / 'bench/spider/data/spider'
        if not root.exists():
            raise AssertionError(
                'spider dataset not found follow the instruction in bench/README'
            )
        self.schemas = load_schemas(
            schemas_path=root / 'tables.json', db_path=root / 'database'
        )
        self.evaluator = Evaluator(root)
        self.evaluate = self.evaluator.evaluate
        self.dev_data = [
            SpiderExample(self, x) for x in load_spider_data(root / 'dev.json')
        ]
        self.train_data = [
            SpiderExample(self, x) for x in load_spider_data(root / 'train_spider.json')
        ]


class SpiderExample:
    def __init__(self, interface, example):
        self.interface = interface
        self.gold_sql = example.query
        self.text = example.utterance
        self.db_name = example.schema_name
        self.db_schema = interface.schemas[example.schema_name]

    def evaluate(self, predicted_sql):
        "Is the predicted_sql correct?"
        return self.interface.evaluate(self.gold_sql, predicted_sql, self.db_name)[0]

    def __repr__(self):
        return f'<SpiderExample db={self.db_name!r}, text={self.text!r}>'

    def describe_schema(self):
        table_strs = []
        for table in self.db_schema.tables:
            column_strs = []
            for column in table.columns:
                column_strs.append(
                    f'* {column.name} ({column.tpe.value}): {column.nl_name}'
                )
            table_str = '\n'.join([table.name] + column_strs)
            table_strs.append(table_str)
        return '\n\n'.join(table_strs)

    def describe(self):
        return f"""
You are a coding assistant helping an analyst answer questions over business data in SQL.
More specifically, the analyst provides you a database schema
(tables in the database along with their column names and types)
and asks a question about the data that can be solved by issuing a SQL query to the database.
In response, you write the SQL statement that answers the question.
You do not provide any commentary or explanation of what the code does,
just the SQL statement ending in a semicolon.

Here is a database schema:

{self.describe_schema()}

Please write me a SQL statement that answers the following question: {self.text}

Remember, DO NOT provide any commentary or explanation of what the code does, just the SQL statement ending in a semicolon.
"""

    def run_query(self, query):
        db = self.interface.evaluator.db_path / self.db_name / (self.db_name + '.sqlite')
        with sqlite3.connect(db) as conn:
            df = pd.read_sql_query(query, conn)
        return df


def test_interface():
    spider = SpiderInterface()
    for x in spider.dev_data[:5]:
        print()
        print(x)
        print(x.gold_sql)
        assert x.evaluate(x.gold_sql)


if __name__ == '__main__':
    test_interface()
