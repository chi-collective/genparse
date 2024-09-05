import vllm
from vllm import SamplingParams
import concurrent.futures as cfuts

import ds1000.execution as execution

import os
os.environ["HF_TOKEN"] = "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"
llm = vllm.LLM(
    model="meta-llama/Meta-Llama-3-8B",
    enable_prefix_caching=True,
)

# disable tensorflow logging and no GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TOKENIZERS_PARALLELISM"] = "(true | false)"

from datasets import load_dataset
import pandas as pd
df = pd.DataFrame(load_dataset("xlangai/DS-1000")["test"])
df["library"] = df["metadata"].map(lambda x: x["library"])

executor = cfuts.ProcessPoolExecutor(max_workers=16)

def get_runnable_code_context(code_context):
    code_context_lines = code_context.split('\n')
    new_code_context_lines = []
    for line in code_context_lines:
        if "assert exec_test" in line:
            continue
        else:
            new_code_context_lines.append(line)
    return '\n'.join(new_code_context_lines)

def run_partial_answer(answer: str, code_context: str, completion_id=None):
    test_program = (
        code_context + '\n'
        + f'code = {repr(answer)}\n'
        + 'test_execution(code)\n'
        + ('test_string(code)\n'  if 'test_string(' in code_context  else '\n')
    )
    fut = executor.submit(execution.check_correctness, test_program, timeout=2, completion_id=completion_id)
    return fut

df["runnable_code_context"] = df["code_context"].map(get_runnable_code_context)


n_particles = 10
ess_threshold = 0.5
sampling_params = SamplingParams(
    max_tokens=1024,
    temperature=1.0,
    top_p=1.0,
    n=1,
    stop=["</code>", "END SOLUTION", "\n"],
)
initial_sampling_params = SamplingParams(
    max_tokens=1024,
    temperature=1.0,
    top_p=1.0,
    n=10,
    stop=["</code>", "END SOLUTION", "\n"],
    seed=0,
)

row = df.iloc[3]
prompt = row["prompt"]

class Particle:
    def __init__(self, output):
        # output: vllm.CompletionOutput
        self.patience = 5
        self.outputs = [output]
        self.context = output.text
        self.cumulative_logprob = output.cumulative_logprob
        self.weight = self.cumulative_logprob
        self.done = output.stop_reason != "\n"

    def update(self, output):
        if self.done:
            raise ValueError("cannot update a finished particle")
        self.outputs.append(output)
        self.context += "\n" + output.text
        self.cumulative_logprob += output.cumulative_logprob
        self.weight += self.cumulative_logprob
        self.done = output.stop_reason != "\n"


def get_initial_particles(prompt):
    # a particle is a list of llm_outputs, do a first step to initialize the particles
    llm_outputs = llm.generate(prompt, sampling_params=initial_sampling_params)
    return [Particle(output) for output in llm_outputs[0].outputs]


def step_particles(particles, prompt):
    prompts = []
    prompt_ids = []
    for i, p in enumerate(particles):
        if not p.done:
            prompts.append(prompt + p.context)
            prompt_ids.append(i)
    llm_outputs = llm.generate(prompts, sampling_params)
    for i, output in zip(prompt_ids, llm_outputs):
        particles[i].update(output.outputs[0])


def print_particles(particles):
    print("=" * 120)
    for p in particles:
        print(p.context)
        print("-" * 100)
    print("=" * 120)

import numpy as np
from scipy.special import logsumexp
from copy import deepcopy
from pprint import pprint

particles = get_initial_particles(prompt)
print_particles(particles)
while not all(particle.done for particle in particles):
    futs = [run_partial_answer(p.context, row["runnable_code_context"]) for p in particles]
    pprint([(particle.context, particle.patience, fut.result()) for particle, fut in zip(particles, futs)])

    log_weights = np.array([p.cumulative_logprob for p in particles])
    has_unrunnable_code = False
    for i, fut in enumerate(futs):
        if fut.result()["result"] != "syntax error" and not fut.result()["passed"]:
            log_weights[i] = -np.inf
            has_unrunnable_code = True
        elif fut.result()["result"] == "syntax error":
            particles[i].patience -= 1
        elif fut.result()["result"] == "passed":
            particles[i].patience = 5
        elif "import pandas" in particles[i].context:
            log_weights[i] = -np.inf
            has_unrunnable_code = True

        if particles[i].patience <= 0:
                log_weights[i] = -np.inf
                has_unrunnable_code = True
                # if resampling happens too often,
                # it will create a situation where all particles get stuck,
                # and we wont discover multiline solutions

    print(log_weights)
    log_total = logsumexp(log_weights)
    log_normalized_weights = log_weights - log_total
    log_ess = -logsumexp(2 * log_normalized_weights)
    if has_unrunnable_code or np.exp(log_ess) < ess_threshold:
        print("resampling")
        print("=" * 100)
        resampled_indices = np.random.choice(np.arange(n_particles), size=n_particles, p=np.exp(log_weights - log_total))
        particles = [deepcopy(particles[i]) for i in resampled_indices]
        avg_weight = log_total - np.log(n_particles)
        for p in particles:
            p.weight = avg_weight

    step_particles(particles, prompt)
    print_particles(particles)












