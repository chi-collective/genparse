import numpy as np
from collections import namedtuple, Counter
from arsenal.datastructures.pdict import pdict
from arsenal.iterextras import head_iter
from genparse.proposal import TokenProposal


from time import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn


Item = namedtuple('Item', 'ps, xs, ys')


class Crunching:
    """A* enumeration of sequences subject to constraints.

    Crunching (N-Best approximation) is a technique used to marginalize over
    latent variables (typically aligments) in machine translation.  People often
    credit May and Knight (2006) in the context of machine translation.

    """

    def __init__(self, *, llm, guide):
        self.llm = llm
        self.guide = guide
        self.T = TokenProposal(llm=llm, guide=guide, K=None)

    def posterior_enumerate(
        self,
        prompt,
        depth=float('inf'),
        beam_width=float('inf'),
        max_generations=float('inf'),
    ):
        """Enumerate strings in the posterior high probability first.

        Optional parameters:

         - `depth`: truncate the posterior to strings that are `<= depth` tokens long.

         - `beam_width`: Prioritized strings that are all within the
           top-(`beam_width`) for all lengths over other strings that may be
           more probable otherwise.

         - `max_generations` will continue enumerating strings that would have
           survived the top-`beam_width` filtering for `max_generations` rounds.

        """
        Q = pdict()

        n_bucket = Counter()

        it = head_iter(self._iter_p_next(Item(1, (), prompt)))
        Q[it] = (0, -1)

        curr_priority = (0, float('-inf'))

        start_time = time()

        console = Console()
        with Progress(
            SpinnerColumn(),
            TextColumn('[progress.description]{task.description}', style='bold magenta'),
            TextColumn(
                'priority: {task.fields[priority]}, {task.fields[rate]:.2f} nodes/sec ({task.fields[nodes]} nodes / {task.fields[elapsed]:.2f} sec)'
            ),
            console=console,
        ) as progress:
            task = progress.add_task(
                'Search', nodes=0, priority=None, elapsed=0.0, rate=0
            )

            nodes = 0
            while Q:
                nodes += 1
                iterator, p = Q.popitem()

                (generation, _) = p

                # priority must be ascending
                assert curr_priority <= p, [curr_priority, p]
                curr_priority = max(curr_priority, p)

                if generation >= max_generations:
                    break

                if iterator.done:
                    continue
                item = next(iterator)

                bucket = (len(item.xs), generation)
                n_bucket[bucket] += 1

                progress.update(
                    task,
                    nodes=nodes,
                    priority=f'({p[0]}, {p[1]:.2f})',
                    elapsed=time() - start_time,
                    rate=nodes / (time() - start_time),
                )

                if item.xs[-1] == self.T.new_eos:
                    if self.guide(''.join(item.xs)):
                        yield item
                    continue

                if n_bucket[bucket] > beam_width:
                    advance = 1
                else:
                    advance = 0

                if len(item.xs) < depth:
                    # Extend `item` by an additional token; creates an new iterator
                    extend_iter = head_iter(self._iter_p_next(item))
                    if not extend_iter.done:
                        Q[extend_iter] = (
                            generation + advance,
                            -np.log(extend_iter.head.ps),
                        )

                if not iterator.done:
                    # Put the iterator back on the queue
                    Q[iterator] = (generation + advance, -np.log(iterator.head.ps))

    def _iter_p_next(self, item):
        """
        This method will lazily enumerate the nodes in the intersection of `llm` and
        and the `guide` for the given context using the TokenProposal.
        """
        for token, value in self.T.traverse_trie(item.xs, self.llm.p_next(item.ys)):
            yield Item(
                item.ps * value,
                item.xs + (token,),
                item.ys + ((token,) if token != self.T.new_eos else (self.T.old_eos,)),
            )
