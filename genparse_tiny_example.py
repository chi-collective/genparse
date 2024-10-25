"""
GenParse Tiny Example

This script demonstrates a simple use case of the GenParse library for constrained text generation.
It uses a basic grammar to generate completions for the phrase "Sequential Monte Carlo is",
constraining the output to either "good" or "bad".

The script showcases how to set up inference, run it, and process the results to obtain
probabilities for each generated text.
"""

from genparse import InferenceSetup


def main():
    print('Starting GenParse tiny example...')

    # Define a simple grammar using Lark syntax
    grammar = """
    start: "Sequential Monte Carlo is " ( "good" | "bad" )
    """
    print('Grammar defined.')

    # Initialize InferenceSetup with GPT-2 model and character-level proposal
    inference_setup = InferenceSetup('gpt2', grammar, proposal_name='character')
    print('InferenceSetup created successfully.')

    # Run inference with a single space as the inital prompt, 5 particles,
    # and set verbosity to 1 to print progress to the console
    inference_result = inference_setup(' ', n_particles=5, verbosity=1)
    print('Inference completed.')

    # Display probabilities using the .posterior property
    print('Probabilities using the .posterior property:')
    posterior = inference_result.posterior

    # Display results
    print(posterior)


if __name__ == '__main__':
    main()
