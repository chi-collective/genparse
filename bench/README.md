# Benchmarking

## Spider

### Setup

To download the Spider dataset, I use the `gdown` package, which is available on PyPI:

```bash
pip install gdown
```

assuming you're currently in the `bench` directory, do

```bash
cd spider/data
gdown 'https://drive.google.com/u/0/uc?id=1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m&export=download'
unzip spider.zip
```

also, download the evaluation codebase:

```bash
git clone https://github.com/taoyds/spider.git spider-eval
```

before running any evaluation, `spider-eval` depends on `punkt` package of `nltk, so download that first:

```
  >>> import nltk
  >>> nltk.download('punkt')
```

### Generation

to generate sql on the first 100 examples on the spider dev set on llama-2-chat-13b model, from the root directory, do
```bash
python bench/run_spider_llama2_chat.py --n-query 100 --model-size 13b --exp-name 13b-100
```

which will generated `spider-eval/gold-13b-100.txt` and `spider-eval/predicted-13b-100.txt`.

### Evaluation

you need to run evaluation inside the eval codebase: 

```bash
cd spider-eval
```

Then, for example, to evaluate the generated sqls from above (`13b-100`), do

```bash
exp_name="13b-100"
python evaluation.py \
  --db ../bench/spider/data/spider/database \
  --table ../bench/spider/data/spider/tables.json \
  --etype "all" \
  --gold gold-${exp_name}.txt --pred predicted-${exp_name}.txt
```
