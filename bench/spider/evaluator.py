from pathlib import Path

import bench.spider.evaluation as E
from bench.spider.evaluation import (
    build_foreign_key_map_from_json,
    build_valid_col_units,
    rebuild_sql_val,
    rebuild_sql_col,
    eval_exec_match,
)
import sqlite3

class Evaluator:
    def __init__(self, spider_dir: Path):
        self.tables_path = spider_dir / 'tables.json'
        self.db_path = spider_dir / 'database'
        self.kmaps = build_foreign_key_map_from_json(self.tables_path)

    def get_eval(self, pred: str, db_name: str):
        """Returns: bool, Optional[str]

        On success (i.e., predicted execution result is the same as gold), returns `(True, None)`
        On failure, returns `(False, reason)` where reason is one of the two cases:
        * `invalid` if `pred` sql is not a well-formed sql statement that can be parsed by sqlite
        * `mismatch` if `pred` is a well-formed sql but the execution result is different from that of the `gold`.
        """
        db = self.db_path / db_name / (db_name + '.sqlite')
        schema = E.Schema(E.get_schema(db))

        try:
            p_sql = E.get_sql(schema, pred)
        except Exception:
            # sql is ill-formed (can't be parsed by sqlite engine)
            return None

        kmap = self.kmaps[db_name]

        p_valid_col_units = build_valid_col_units(p_sql['from']['table_units'], schema)
        p_sql = rebuild_sql_val(p_sql)
        p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)

        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        try:
            cursor.execute(pred)
            p_res = cursor.fetchall()
        except Exception:
            return None

        def res_map(res, val_units):
            rmap = {}
            for idx, val_unit in enumerate(val_units):
                key = (
                    tuple(val_unit[1])
                    if not val_unit[2]
                    else (val_unit[0], tuple(val_unit[1]), tuple(val_unit[2]))
                )
                rmap[key] = [r[idx] for r in res]
            return rmap

        p_val_units = [unit[1] for unit in p_sql['select'][1]]
        return res_map(p_res, p_val_units)

    def evaluate(self, gold: str, pred: str, db_name: str):
        """Returns: bool, Optional[str]

        On success (i.e., predicted execution result is the same as gold), returns `(True, None)`
        On failure, returns `(False, reason)` where reason is one of the two cases:
        * `invalid` if `pred` sql is not a well-formed sql statement that can be parsed by sqlite
        * `mismatch` if `pred` is a well-formed sql but the execution result is different from that of the `gold`.
        """
        db = self.db_path / db_name / (db_name + '.sqlite')
        schema = E.Schema(E.get_schema(db))
        g_sql = E.get_sql(schema, gold)

        try:
            p_sql = E.get_sql(schema, pred)
        except Exception:
            # sql is ill-formed (can't be parsed by sqlite engine)
            return False, 'invalid'

        kmap = self.kmaps[db_name]
        g_valid_col_units = build_valid_col_units(g_sql['from']['table_units'], schema)
        g_sql = rebuild_sql_val(g_sql)
        g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)
        p_valid_col_units = build_valid_col_units(p_sql['from']['table_units'], schema)
        p_sql = rebuild_sql_val(p_sql)
        p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)

        exec_match = eval_exec_match(db, pred, gold, p_sql, g_sql)
        reason = None if exec_match else 'mismatch'

        return exec_match, reason