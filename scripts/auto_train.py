import subprocess
import sys

CYCLES = 6  # change this to however many cycles you want

for i in range(1, CYCLES + 1):
    print(f"\n{'='*50}")
    print(f"CYCLE {i} of {CYCLES}")
    print(f"{'='*50}")

    # Step 1 - run backtest
    print(f"\n--- Running backtest ---")
    subprocess.run([sys.executable, "scripts/run_backtest.py", 
                   "--config", "config.yaml"], check=True)

    # Step 2 - train model
    print(f"\n--- Training model ---")
    subprocess.run([sys.executable, "scripts/train_regression.py",
                   "--config", "config.yaml"], check=True)

print(f"\n{'='*50}")
print("✅ All cycles complete!")
print("Check mean R trend above to see improvement.")
print(f"{'='*50}")