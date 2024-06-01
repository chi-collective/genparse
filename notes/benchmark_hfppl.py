import sys
import os
import logging
logger = logging.getLogger(__name__)

# COMMENT THESE OUT IF YOU ARE NOT ME
#sys.path.append("/home/mila/b/benjamin.lebrun/genparse")
#os.environ["HF_HOME"] = os.path.join(os.environ["SCRATCH"], "hf_cache")

from hfppl import Model, CachedCausalLM, LMContext, smc_standard, smc_steer
from hfppl.distributions import TokenCategorical
from transformers import AutoTokenizer, AutoModel

import genparse
from genparse.cfglm import BoolMaskCFGLM, EarleyBoolMaskCFGLM
from genparse.util import LarkStuff
from genparse import EOS, Float
from arsenal.maths import sample_dict, logsumexp
from genparse.proposal import CharacterProposal
from genparse.lm import LM
from arsenal import timers

def main():
    logging.basicConfig(filename='hfppl_benleb.log', level=logging.INFO)

    logger.info("Loading model and tokenizer")

    MODEL_ID = "codellama/CodeLlama-7b-Instruct-hf"
    hfppl_llm = CachedCausalLM.from_pretrained(MODEL_ID, load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, 
        use_fast=True,
        prefix_token=None, 
        middle_token=None, 
        suffix_token=None, 
        eot_token=None, 
        fill_token=None
    )

    logger.info("Creating grammar")

    prompt = """
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
    Q: Write a SQL query that returns white voters' average age for each state color. 
    A:"""

    guide = EarleyBoolMaskCFGLM(
        LarkStuff(
            r"""
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
        ).char_cfg(.99, ignore='[ ]?')
    )

    import numpy as np
    import time
    import asyncio
    from genparse import Float

    class SteeringModel(Model):
        def __init__(self, llm, guide, proposal, prompt, max_tokens, compare_time=False):
            super().__init__()
            self.llm = llm # GreedilyTokenizedLM
            self.guide = guide # PCFGLM
            self.prompt = prompt
            self.context = ""
            self.proposal = proposal # CharacterProposal
            self.max_tokens = max_tokens
            self.compare_time = compare_time

        async def step(self):
            (token, llm_prob, guide_prob, proposal_prob) = await self.proposal.sample_next_token(
                context=self.context, prompt=self.prompt, compare_time=self.compare_time
            )
            self.context += token
            self.weight += np.log(llm_prob) + np.log(guide_prob) - np.log(proposal_prob)
            self.max_tokens -= 1

            print(f"Sampled token=`{token}`. Particle={self.context}")

            if token == self.llm.eos or self.max_tokens == 0 or token == genparse.EOS:
                self.finish()
                return

        def immutable_properties(self):
            return ['llm', 'prompt', 'guide', 'compare_token']


    class GreedilyTokenizedLLM:
        def __init__(self, llm, tokenizer):
            self.tokenizer = tokenizer
            self._model = llm # hfppl Model
            self._decode = [self.tokenizer.decode([i]) for i in range(self.tokenizer.vocab_size)]
            self.V = set(self._decode)
            self.eos = self.tokenizer.eos_token

        def __call__(self, xs):
            return self.model(self.tokenizer.encode(xs))

        async def p_next(self, xs, top=None):
            return await self._p_next(xs, top=top)

        async def _p_next(self, xs, top=None):
            assert isinstance(xs, str)
            tokens = self.tokenizer.encode(xs)

            _logp = await self._model.next_token_logprobs(tokens)
            _p = np.exp(_logp)

            if top is None:
                top_p = _p.argsort()
            else:
                top_p = _p.argsort()[-top:]
            pp = Float.chart()
            for i in reversed(top_p):
                pp[self._decode[i]] = _p[i]
            if top is None:
                return pp
            else:
                return pp.normalize()

    logger.info("Running SMC")

    genparse_llm = GreedilyTokenizedLLM(hfppl_llm, tokenizer)
    proposal = CharacterProposal(llm=genparse_llm, guide=guide)
    steering_model = SteeringModel(genparse_llm, guide, proposal, prompt, 100, compare_time=False)
    particles = asyncio.run(smc_standard(steering_model, n_particles=3))
    
    '''
    LLM = hfppl_llm

    class PureModel(Model):
        def __init__(self, prompt, max_tokens):
            super().__init__()
            self.context = LMContext(LLM, prompt)
            self.max_tokens = max_tokens
            
        async def step(self):
            logprobs = self.context.next_token_logprobs.copy()
            proposal = TokenCategorical(LLM, logprobs)     
            
            token = await self.sample(self.context.next_token(), proposal=proposal)
            #self.score(logprobs[token.token_id])
            self.max_tokens -= 1
            
            # Check if done
            if token == LLM.tokenizer.eos_token_id or self.max_tokens == 0:
                self.finish()
                return
    max_tokens = 50
    constraint_model = PureModel(prompt, max_tokens=max_tokens)
    particles = asyncio.run(smc_standard(constraint_model, 5))
    '''
if __name__ == "__main__":
    main()
