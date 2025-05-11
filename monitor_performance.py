import time
import subprocess

def evaluate_performance(target, *args, mode='function', runs=1):
    """
    Evaluate the runtime performance of a Python function or script file multiple times and return the average time.

    Parameters:
        target (callable or str): If mode='function', this should be a function to run.
                                  If mode='file', this should be the path to a .py file.
        *args: Arguments to pass to the function if running in 'function' mode.
        mode (str): 'function' to evaluate a function, 'file' to evaluate a Python script file.
        runs (int): Number of times to run the performance evaluation. Defaults to 1 (single run).

    Returns:
        float: The average execution time over the specified number of runs.

    Example usage:
        # For function evaluation
        def sample_function():
            sum([x**2 for x in range(1000000)])

        print(evaluate_performance(sample_function, mode='function', runs=10))

        # For script evaluation
        print(evaluate_performance('example_script.py', mode='file', runs=5))
    """
    if mode not in ['function', 'file']:
        raise ValueError("Mode must be either 'function' or 'file'.")

    total_time = 0

    for _ in range(runs):
        start_time = time.time()

        if mode == 'function':
            if not callable(target):
                raise ValueError("In 'function' mode, target must be a callable function.")
            target(*args)

        elif mode == 'file':
            subprocess.run(["python", target], check=True)

        end_time = time.time()
        total_time += end_time - start_time

    average_time = total_time / runs
    print(f"Average execution time over {runs} runs: {average_time:.4f} seconds")
    return average_time
