from genparse import InferenceSetup
import math


def main():
    print('Starting GenParse sql example...')

    # Define a simple grammar using Lark syntax
    grammar = """
    start: WS? "SELECT" WS column WS from_clause (WS group_clause)?
    from_clause: "FROM" WS table
    group_clause: "GROUP BY" WS column
    column: "age" | "name" | "salary" | "department"
    table: "employees"
    WS: " "
    """
    print('Grammar defined.')

    # Define a potential function to steer SQL generation
    def potential_function(particles):
        # Example: Prefer queries that group by 'name'
        potentials = []
        for particle in particles:
            context_str = ''.join(
                particle.context
            )  # Combine the context tuple into a string
            if 'name' in context_str:
                potentials.append(-100)  # Slightly disfavored
            elif 'age' in context_str:
                potentials.append(-100)  # Slightly more disfavored
            else:
                potentials.append(0)  # log (1) = 0, neutral score
        return potentials

    # Initialize InferenceSetup with GPT2 model and character-level proposal
    inference_setup = InferenceSetup('gpt2', grammar, proposal_name='character')
    print('InferenceSetup created successfully.')

    # Run inference with a single space as the inital prompt, 5 particles,
    # and set verbosity to 1 to print progress to the console
    # limit tokens to 25 to ensure we return in a timely manner.
    inference_result = inference_setup(
        'Write an SQL query:',
        n_particles=5,
        verbosity=1,
        max_tokens=25,
        return_record=True,
        potential=potential_function,
    )
    print('Inference completed.')

    # Display probabilities using the .posterior property
    print('Probabilities using the .posterior property:')
    posterior = inference_result.posterior

    # Display results
    print(posterior)


if __name__ == '__main__':
    main()
