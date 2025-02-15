import os
import pickle
from argparse import ArgumentParser

import numpy as np
import torch
import transformers
from arsenal import colors

from genparse import Float
from genparse.cfglm import BoolCFGLM
from genparse.lm import TokenizedLLM
from genparse.backends.vllm import vllmpplLLM, VLLMSampler
from genparse.proposal import CharacterProposal, TokenProposal
from genparse.util import set_seed, lark_guide


torch.backends.cuda.matmul.allow_tf32 = True

p = ArgumentParser()
p.add_argument('--model', choices=['gpt2', 'codellama'], required=True)
p.add_argument('--proposal', choices=['token', 'character'], default='character')
p.add_argument('--particles', type=int, default=1)
p.add_argument('--n-beam', type=int, default=1)
p.add_argument('--reps', type=int, default=1)
p.add_argument('--max-tokens', type=int, default=100)
p.add_argument('--verbosity', type=int, default=0)
p.add_argument('--seed', type=int, default=0)
p.add_argument(
    '--inference',
    choices=['smc-standard', 'smc-steer', 'importance-sampling'],
    default='smc-standard',
)
args = p.parse_args()


set_seed(args.seed)


if args.model == 'gpt2':
    MODEL_ID = 'gpt2'
    hfppl_llm = vllmpplLLM(MODEL_ID)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)

elif args.model == 'codellama':
    assert torch.cuda.is_available()

    MODEL_ID = 'codellama/CodeLlama-7b-Instruct-hf'
    hfppl_llm = vllmpplLLM(MODEL_ID, dtype=torch.float32, max_model_len=4096)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)

else:
    raise ValueError(args.model)


prompt_template = """
You have access to a political survey data table named "data", which includes the following columns:
- "age" (integer)
- "gender" ("male" or "female"),
- "year" (integer)
- "state_color" ("blue" or "red")
- "zipcode" (integer)
- "vote" ("democrat" or "republican")
- "registered_party" ("democrat" or "republican")
- "race_ethnicity" ("white", "black", or "latino").

Q: Write a SQL query that shows individuals' age and gender, for people over 50 years old.
A: SELECT age, gender FROM data WHERE age>50 </s>
Q: Write a SQL query that shows individuals' vote and zipcode, ordered from lowest to highest age.
A: SELECT vote, zipcode, age FROM data ORDER BY age ASC </s>
Q: %s
A:"""

grammar = r"""

start: WS? "SELECT" WS select_expr WS "FROM" WS from_expr [WS "WHERE" WS bool_condition] [WS "GROUP BY" WS var_list] [WS "ORDER BY" WS orderby_expr] WS EOS
EOS: "</s>"
select_expr: STAR | select_list
bool_condition: bool_expr | "(" bool_condition WS "AND" WS bool_condition ")" | "(" bool_condition WS "OR" WS bool_condition ")"
bool_expr: var "=" value | var ">" value | var "<" value
from_expr: "data"
orderby_expr: var_list WS "ASC" | var_list WS "DESC"
select_list: select_var ("," WS select_var)*
var_list: var ("," WS var)*
select_var: var | "AVG(" var ")" | "MEDIAN(" var ")" | "COUNT(" var ")"
var: "age" | "gender" | "year" | "state_color" | "zipcode" | "vote" | "race_ethnicity"
value: NUMBER | "'red'" | "'blue'" | "'white'" | "'black'" | "'latino'" | "'republican'" | "'democrat'" | "'male'" | "'female'"
STAR: "*"
NUMBER: /\d+/
WS: /[ \n\r\t]+/

"""


prompts = [
    "Write a SQL query that returns white voters' average age for each state color and sort the results.",
    'Write a SQL query that shows the young republicans.',
    'Write a SQL query that shows the old democrats in Williamsburg.',
    'Write a SQL query that shows the oldest democrat in each red state.',
    'Write a SQL query that shows the average age of red states vs blue states.',
]


def main():
    guide = lark_guide(grammar)

    BATCH_SIZE = 80

    hfppl_llm.batch_size = BATCH_SIZE
    genparse_llm = TokenizedLLM(
        model=hfppl_llm, tokenizer=tokenizer, batch_size=BATCH_SIZE
    )

    sampler = VLLMSampler(llm=genparse_llm, guide=guide)
    if args.proposal == 'character':
        proposal = CharacterProposal(llm=genparse_llm, guide=guide)
    elif args.proposal == 'token':
        proposal = TokenProposal(llm=genparse_llm, guide=guide, K=5)
    else:
        raise ValueError(f'invalid proposal name {args.proposal!r}')

    for _ in range(args.reps):
        for sql_prompt in prompts:
            prompt = prompt_template % sql_prompt
            print(colors.cyan % colors.line(100))
            print(colors.cyan % sql_prompt)

            particles = sampler.run_inference(
                prompt=prompt,
                proposal=proposal,
                method=args.inference,
                n_particles=args.particles,
                max_tokens=args.max_tokens,
                n_beam=args.n_beam,
                verbosity=args.verbosity,
            )

            # if args.particles > 1 and record is not None:
            #     fig = record.plot_particles_trajectory()
            #     fig.write_html('viz.html')
            #     print('wrote to viz.html')

            print(colors.yellow % 'character posterior')
            posterior = Float.chart()
            for p in particles:
                posterior[''.join(p.context).strip()] += np.exp(p.weight)
            print(posterior.normalize())

            if 0:
                print(colors.yellow % 'token posterior')
                posterior = Float.chart()
                for p in particles:
                    posterior[tuple(p.context)] += np.exp(p.weight)
                print(posterior.normalize())

    sampler.timer.plot_feature('t')

    if not os.path.exists('benchmark/results'):
        os.makedirs('benchmark/results')
    file_name = f'benchmark/results/vllm_runtime_{args.model}_{args.max_tokens}_{args.proposal}_{args.inference}_{args.particles}_{args.n_beam}'
    with open(f'{file_name}.pkl', 'wb') as f:
        pickle.dump(sampler.timer, f)
    print(f'wrote to {file_name}.pkl')

    import pylab as pl

    pl.title(args)
    pl.xlabel('context size (characters)')
    pl.savefig(f'{file_name}.pdf')
    print(f'wrote to {file_name}.pdf')
    pl.show()

    # from arsenal.debug import ip
    # ip()


if __name__ == '__main__':
    main()
