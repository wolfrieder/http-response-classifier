import sys
sys.path.append("src")
from src.transformer import parse_raw_data
# Import other transformer scripts here


def run_parse_raw_data(browser, date, filename):
    parse_raw_data.run(browser, date, filename)


# Define other run functions for other transformer scripts here

if __name__ == "__main__":
    script_to_run = sys.argv[1]
    args = sys.argv[2:]

    print(script_to_run)

    if script_to_run == "parse_raw_data.py":
        run_parse_raw_data(*args)
    # Add conditions for other transformer scripts here
    else:
        print(f"Unknown script: {script_to_run}")
