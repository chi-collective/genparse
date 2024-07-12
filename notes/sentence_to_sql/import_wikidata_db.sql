DROP TABLE IF EXISTS People;
PRAGMA encoding = 'utf-8';

.mode csv
.import $csv_path People
.schema People