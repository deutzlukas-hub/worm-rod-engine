from pathlib import Path
import unittest

def main():
    test_dir = Path(__file__).resolve().parent
    test_pattern = 'test_*.py'

    loader = unittest.TestLoader()
    tests = loader.discover(start_dir=str(test_dir), pattern=test_pattern)

    # Create a test runner
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(tests)

if __name__ == '__main__':
    main()
