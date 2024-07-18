import asyncio
from arsenal import colors
from arsenal.maths import sample_dict


class Proposal:
    def sample_next_token_sync(self, *args, **kwargs):
        "Synchronous version of `sample_next_token`."
        return asyncio.run(self.sample_next_token(*args, **kwargs))

    def sample(self, prompt=(), max_tokens=float('inf'), verbosity=0, draw=sample_dict):
        context = ()
        W = 1
        P = 1
        t = 0
        while True:
            t += 1
            if t <= max_tokens:
                (token, proposal_p, weight_update) = self.sample_next_token_sync(
                    prompt=prompt,
                    context=context,
                    draw=draw,
                )
            else:
                token = self.guide.eos
                weight_update = 1
                proposal_p = 1
            W *= weight_update
            P *= proposal_p
            if self.guide.eos == token:
                break
            if verbosity > 0:
                print(colors.cyan % token, end=colors.magenta % '|')
            context = context + (token,)
        if verbosity > 0:
            print()
        return (context, P, W)
