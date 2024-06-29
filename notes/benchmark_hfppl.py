import argparse

PROMPT = """
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

VERY_RESTRICTED_SQL = r"""
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


def main(
    model_name, proposal_name, batch_size, n_particles, method, max_tokens, verbosity
):
    from genparse.cfglm import BoolMaskCFGLM
    from genparse.lm import AsyncGreedilyTokenizedLLM
    from genparse.proposal import CharacterProposal, TokenProposal
    from genparse.steer import HFPPLSampler
    from genparse.util import LarkStuff

    genparse_llm = AsyncGreedilyTokenizedLLM.from_name(model_name, batch_size=batch_size)
    guide = BoolMaskCFGLM(LarkStuff(VERY_RESTRICTED_SQL).char_cfg(0.99, ignore='[ ]?'))
    sampler = HFPPLSampler(llm=genparse_llm, guide=guide)
    if proposal_name == 'character':
        proposal = CharacterProposal(llm=genparse_llm, guide=guide)
    elif proposal_name == 'token':
        proposal = TokenProposal(llm=genparse_llm, guide=guide)
    else:
        ValueError('invalid proposal name')

    sampler.run_inference(
        prompt=PROMPT,
        proposal=proposal,
        method=method,
        n_particles=n_particles,
        max_tokens=max_tokens,
        verbosity=verbosity,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test HFPPL inference.')
    parser.add_argument(
        '--model_name',
        type=str,
        default='codellama/CodeLlama-7b-Instruct-hf',
        help='huggingface model path',
    )
    parser.add_argument(
        '--proposal', type=str, default='character', help='`character` or `token`'
    )
    parser.add_argument(
        '--batch_size', type=int, default=20, help='batch size for inference'
    )
    parser.add_argument(
        '--method', type=str, default='smc-standard', help='smc-standard or smc-steer'
    )
    parser.add_argument('--n_particles', type=int, default=5, help='Number of particles')
    parser.add_argument('--max_tokens', type=int, default=50, help='Maximum tokens')
    parser.add_argument(
        '--verbosity',
        type=int,
        default=1,
        help='Verbosity = 1 prints tokens at each time-step; 0 is silent',
    )

    args = parser.parse_args()
    main(
        args.model_name,
        args.proposal,
        args.batch_size,
        args.n_particles,
        args.method,
        args.max_tokens,
        args.verbosity,
    )
