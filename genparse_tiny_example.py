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

    # Run inference with 15 particles
    inference_result = inference_setup(' ', n_particles=10)
    print('Inference completed.')

    # Process the results and display probabilities
    process_and_display_results(inference_result.particles)


def process_and_display_results(particles):
    """
    Process particles, calculate probabilities, and display results.

    Args:
        particles: Iterable of Particle objects from InferenceSetup result.
    """
    # Create a dictionary to store unique texts and their total weights
    text_weights = {}

    for particle in particles:
        generated_text = ''.join(particle.context)
        particle_weight = particle.weight
        if generated_text in text_weights:
            text_weights[generated_text] += particle_weight
        else:
            text_weights[generated_text] = particle_weight

    # Calculate the total weight of all particles
    total_weight = sum(text_weights.values())

    # Display results
    print('{')
    for generated_text, weight in text_weights.items():
        probability = weight / total_weight
        print(f"  '{generated_text}': {probability},")
    print('}')


if __name__ == '__main__':
    main()
