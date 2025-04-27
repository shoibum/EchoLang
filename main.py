#!/usr/bin/env python
# main.py - Entry point for EchoLang application

import argparse
import sys
import os
import logging
import subprocess
import unittest

# Configure logging (can be more sophisticated later)
# Set default level to INFO, but allow overriding via ENV variable?
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
# Reduce verbosity of httpx logger used by Gradio/HF Hub
logging.getLogger("httpx").setLevel(logging.WARNING)
# Reduce verbosity of HF Hub downloader
logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)


logger = logging.getLogger("EchoLangMain")

def parse_args():
    parser = argparse.ArgumentParser(
        description="EchoLang - Multilingual Speech ↔ Text (Whisper+NLLB+XTTS)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a publicly shareable Gradio link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port number to run the Gradio server on"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run unit tests (requires test files in 'tests/' directory)"
    )
    parser.add_argument(
        "--reset-models",
        action="store_true",
        help="Run the script to remove locally managed models (XTTS)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=log_level, # Default from above
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level for the application"
    )
    return parser.parse_args()

def run_reset_script():
    """Detects OS and runs the appropriate model reset script."""
    logger.info("Executing local model reset script (for XTTS)...")
    script_to_run = ""
    script_executable = [] # Command list for subprocess

    if os.name == 'nt':
        script_to_run = "reset_models.bat" # You'd need to create this BAT file
        script_executable = [script_to_run]
        logger.warning("Windows reset script (.bat) execution is basic. Ensure reset_models.bat exists and works.")
    else:
        script_to_run = "reset_models.sh"
        script_path = Path(script_to_run)
        if not script_path.exists():
             logger.error(f"Model reset script '{script_to_run}' not found!")
             print(f"\nError: Model reset script '{script_to_run}' not found!\n", file=sys.stderr)
             sys.exit(1)
        # Ensure executable
        if not os.access(script_path, os.X_OK):
            logger.warning(f"Script '{script_to_run}' not executable. Attempting 'chmod +x'...")
            try:
                subprocess.run(['chmod', '+x', str(script_path)], check=True)
            except Exception as e:
                 logger.error(f"Failed to make '{script_to_run}' executable: {e}", exc_info=True)
                 print(f"Error: Could not make reset script '{script_to_run}' executable.", file=sys.stderr)
                 sys.exit(1)
        script_executable = [f"./{script_to_run}"] # Relative path execution

    logger.info(f"Running command: {' '.join(script_executable)}")
    try:
         process = subprocess.run(
             script_executable,
             check=True, capture_output=True, text=True, shell=False
         )
         logger.info("Reset script stdout:\n---\n%s---", process.stdout.strip())
         if process.stderr: logger.warning("Reset script stderr:\n---\n%s---", process.stderr.strip())
         logger.info("Model reset script completed successfully.")
         print("✅ Local model reset script completed.")
    except FileNotFoundError:
         logger.error(f"Error: Command '{script_executable[0]}' not found.", exc_info=True)
         print(f"\nError: Could not execute '{script_executable[0]}'. Not found.\n", file=sys.stderr)
         sys.exit(1)
    except subprocess.CalledProcessError as e:
         logger.error(f"Reset script failed (exit code {e.returncode})", exc_info=True)
         logger.error("--- Script stdout ---\n%s", e.stdout)
         logger.error("--- Script stderr ---\n%s", e.stderr)
         print(f"\nError: Model reset script failed (exit code {e.returncode}). Check logs.\n", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
         logger.error(f"Unexpected error running reset script: {e}", exc_info=True)
         print(f"\nError: Unexpected error running reset script: {e}\n", file=sys.stderr)
         sys.exit(1)

def run_tests():
    """Discovers and runs unit tests."""
    logger.info("Attempting to run unit tests...")
    test_dir = "tests"
    if not os.path.isdir(test_dir):
         logger.error(f"Test directory '{test_dir}' not found.")
         print(f"\nError: Test directory '{test_dir}' not found.\n", file=sys.stderr)
         sys.exit(1)

    try:
        logger.info(f"Discovering tests in '{test_dir}'...")
        loader = unittest.TestLoader()
        # Ensure src is importable for tests
        sys.path.insert(0, os.path.abspath('.'))
        tests = loader.discover(test_dir, pattern='test_*.py')

        if tests.countTestCases() == 0:
             logger.warning(f"No tests found in '{test_dir}' matching pattern 'test_*.py'.")
             print("\nWarning: No tests found.\n")
             return # Not a failure if no tests exist

        logger.info(f"Running {tests.countTestCases()} tests...")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(tests)

        if not result.wasSuccessful():
             logger.error("Unit tests failed.")
             print("\n❌ Unit tests failed. Check output above.\n", file=sys.stderr)
             sys.exit(1)
        else:
             logger.info("Unit tests passed successfully.")
             print("\n✅ Unit tests passed.\n")

    except ImportError as e:
         logger.error(f"Could not import 'unittest' or test dependencies: {e}")
         print(f"\nError: Could not import test components: {e}\n", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
         logger.error(f"An error occurred during test execution: {e}", exc_info=True)
         print(f"\nError: An unexpected error occurred during tests: {e}\n", file=sys.stderr)
         sys.exit(1)


def main():
    args = parse_args()

    # Reconfigure logging level if specified by arg
    current_level = logging.getLogger().getEffectiveLevel()
    new_level = logging.getLevelName(args.log_level)
    if new_level != current_level:
        logging.getLogger().setLevel(new_level)
        logger.info(f"Logging level set to: {args.log_level}")

    logger.info(f"Starting EchoLang...")
    logger.debug(f"Parsed arguments: {args}")
    logger.debug(f"Python version: {sys.version}")
    logger.debug(f"Current working directory: {os.getcwd()}")


    if args.reset_models:
        run_reset_script()
        sys.exit(0)

    if args.test:
        run_tests()
        sys.exit(0)

    logger.info("Proceeding to launch Gradio application...")
    try:
        from src.web.app import launch_app
    except ImportError as e:
         logger.error(f"Failed to import application components: {e}", exc_info=True)
         print("\n❌ Error: Failed to import application components.", file=sys.stderr)
         print("   Please ensure dependencies are installed ('./setup.sh') and venv is active.", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
         logger.error(f"Unexpected error during application import: {e}", exc_info=True)
         print(f"\n❌ Error during application import: {e}\n", file=sys.stderr)
         sys.exit(1)

    # Import config here to access device info for log message
    try:
        from src import config
        device_info = f"{config.APP_DEVICE} ({config.APP_TORCH_DTYPE})"
    except Exception:
        device_info = "Unknown"


    print("-" * 30)
    print(f"🚀 Starting EchoLang UI")
    print(f"   Device: {device_info}")
    print(f"   Port: {args.port}")
    if args.share: print("   Public link sharing enabled.")
    print("   Please wait for models to load/download on first use...")
    print("-" * 30)

    launch_app(share=args.share, server_port=args.port)

if __name__ == "__main__":
    main()