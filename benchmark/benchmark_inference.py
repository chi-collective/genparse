import numpy as np
import asyncio#, nest_asyncio; nest_asyncio.apply()
from arsenal import timeit, colors

from random import seed
from torch import manual_seed
from transformers import set_seed

from hfppl import Model, CachedCausalLM
from transformers import AutoTokenizer

import genparse
from genparse.cfglm import EarleyBoolMaskCFGLM
from genparse.util import LarkStuff
from genparse import EOS, Float
from genparse.proposal import CharacterProposal, TokenProposal
from genparse.lm import AsyncGreedilyTokenizedLLM
from genparse.inference import smc_standard
from genparse.steer import HFPPLSampler


from argparse import ArgumentParser
p = ArgumentParser()
p.add_argument('--model', choices=['gpt2', 'codellama'], required=True)
p.add_argument('--proposal', choices=['token', 'character'], default='character')
p.add_argument('--particles', type=int, default=1)
p.add_argument('--reps', type=int, default=1)
p.add_argument('--max-tokens', type=int, default=100)
p.add_argument('--verbosity', type=int, default=0)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--inference', choices=['smc-standard', 'smc-steer', 'importance-sampling'], default='smc-standard')
args = p.parse_args()


RANDOM_SEED = args.seed
set_seed(RANDOM_SEED)
seed(RANDOM_SEED)
manual_seed(RANDOM_SEED)


if args.model == 'gpt2':
    import transformers
    from genparse.lm import LLM
    MODEL_ID = "gpt2"
    hfppl_llm = LLM(transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID))
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)

elif args.model == 'codellama':

    import torch
    assert torch.cuda.is_available()

    MODEL_ID = "codellama/CodeLlama-7b-Instruct-hf"
    hfppl_llm = CachedCausalLM.from_pretrained(MODEL_ID, load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, use_fast=True, eot_token=None, fill_token=None,
        prefix_token=None, middle_token=None, suffix_token=None,
    )

else:
    raise ValueError(args.model)


def main():

    prompt_template = """
You have access to a political survey data table named "data", which includes the following columns:
- "age" (integer)
- "gender" ("male" or "female"),
- "year" (integer)
- "state_color" ("blue" or "red")
- "zipcode" (integer)
- "vote" ("democrat" or "republican")
- "race_ethnicity" ("white", "black", or "latino").

Q: Write a SQL query that shows individuals' age and gender, for people over 50 years old.
A: SELECT age, gender FROM data WHERE age>50 </s>
Q: Write a SQL query that shows individuals' vote and zipcode, ordered from lowest to highest age.
A: SELECT vote, zipcode, age FROM data ORDER BY age ASC </s>
Q: %s
A:"""

    sql_prompt = "Write a SQL query that returns white voters' average age for each state color and sort the results."

    sql_prompt = "Write a SQL query that shows the young republicans."

    prompt = prompt_template % sql_prompt

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
    WS: /[ ]/

    """

    character_cfg = LarkStuff(grammar).char_cfg(.99, ignore='[ ]?')

    guide = EarleyBoolMaskCFGLM(character_cfg)

    MAX_TOKENS = 100
    BATCH_SIZE = 80

    hfppl_llm.batch_size = BATCH_SIZE
    genparse_llm = AsyncGreedilyTokenizedLLM(model=hfppl_llm, tokenizer=tokenizer, batch_size=BATCH_SIZE)


    guide = EarleyBoolMaskCFGLM(LarkStuff(grammar).char_cfg(.99, ignore='[ ]?'))
    sampler = HFPPLSampler(llm=genparse_llm, guide=guide)
    if args.proposal == "character":
        proposal = CharacterProposal(llm=genparse_llm, guide=guide)
    elif args.proposal == "token":
        proposal = TokenProposal(llm=genparse_llm, guide=guide, K=5)
    else:
        raise ValueError(f"invalid proposal name {args.proposal!r}")


    Particles = []
    for _ in range(args.reps):

        particles, record = sampler.run_inference(
            prompt=prompt,
            proposal=proposal,
            method=args.inference,
            n_particles=args.particles,
            max_tokens=args.max_tokens,
            verbosity=args.verbosity,
            return_record=True,
        )

        if args.particles > 1 and record is not None:
            fig = record.plot_particles_trajectory()
            fig.write_html("viz.html")
            print('wrote to viz.html')

        print(colors.yellow % 'character posterior')
        posterior = Float.chart()
        for p in particles:
            posterior[''.join(p.context).strip()] += np.exp(p.weight)

        print(posterior.normalize())

        print(colors.yellow % 'token posterior')
        posterior = Float.chart()
        for p in particles:
            posterior[tuple(p.context)] += np.exp(p.weight)

        print(posterior.normalize())

    proposal.timer.plot_feature('t')
    import pickle
    with open('runtime.pkl', 'wb') as f:
        pickle.dump(proposal.timer, f)
    print('wrote to runtime.pkl')

    import pylab as pl
    pl.title(args)
    pl.xlabel('|context| (characters)')
    pl.savefig('runtime.pdf')
    print('wrote to runtime.pdf')
    pl.show()

    #from arsenal.debug import ip
    #ip()


if __name__ == '__main__':
    main()
