#!/usr/bin/env python3
"""
TinyLCM Example Launcher
------------------------
This script makes it easy to run different TinyLCM example scenarios.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def list_scenarios():
    """List all available scenarios."""
    scenario_dir = Path(__file__).parent / "scenarios"
    main_files = list(scenario_dir.glob("main_*.py"))
    scenarios = [f.stem.replace("main_", "") for f in main_files]
    return sorted(scenarios)

def run_scenario(scenario, headless=False, config=None):
    """Run a specific scenario."""
    script_path = Path(__file__).parent / "scenarios" / f"main_{scenario}.py"
    
    if not script_path.exists():
        print(f"Error: Scenario '{scenario}' not found.")
        print("Available scenarios:")
        for s in list_scenarios():
            print(f"  - {s}")
        sys.exit(1)
    
    cmd = [sys.executable, str(script_path)]
    
    if headless:
        cmd.append("--headless")
    
    if config:
        config_path = Path(config)
        if not config_path.exists():
            print(f"Error: Config file '{config}' not found.")
            sys.exit(1)
        cmd.extend(["--config", config])
    
    print(f"Running scenario: {scenario}")
    print(f"Command: {' '.join(cmd)}")
    
    # Make script executable if needed
    script_path.chmod(script_path.stat().st_mode | 0o111)
    
    try:
        process = subprocess.run(cmd)
        return process.returncode
    except KeyboardInterrupt:
        print("Scenario interrupted by user.")
        return 130
    except Exception as e:
        print(f"Error running scenario: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(description="TinyLCM Example Launcher")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available scenarios")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run a specific scenario")
    run_parser.add_argument("scenario", help="Scenario to run", nargs="?", default="autonomous")
    run_parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI")
    run_parser.add_argument("--config", help="Path to custom config file")
    
    args = parser.parse_args()
    
    # Default to "run" if no command is provided
    command = args.command or "run"
    
    if command == "list":
        scenarios = list_scenarios()
        if scenarios:
            print("Available scenarios:")
            for s in scenarios:
                print(f"  - {s}")
        else:
            print("No scenarios found.")
    
    elif command == "run":
        scenario = args.scenario
        return run_scenario(scenario, args.headless, args.config)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())