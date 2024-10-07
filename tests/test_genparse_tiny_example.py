import pytest
import io
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import the main function from genparse_tiny_example
from genparse_tiny_example import main


def test_main_runs_to_completion():
    # Redirect stdout to capture print statements
    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        # Run the main function
        main()
    except Exception as e:
        pytest.fail(f'main() raised an exception: {e}')
    finally:
        # Restore stdout
        sys.stdout = sys.__stdout__

    # Get the captured output
    output = captured_output.getvalue()

    # Check for key phrases that indicate successful completion
    assert 'Starting GenParse tiny example...' in output
    assert 'Grammar defined.' in output
    assert 'InferenceSetup created successfully.' in output
    assert 'Inference completed.' in output
    assert (
        '{' in output and '}' in output
    )  # Check for the presence of the result dictionary


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
