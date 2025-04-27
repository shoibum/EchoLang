#!/usr/bin/env python
# main.py - Entry point for EchoLang application

import argparse
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser(description="EchoLang - Multilingual Speech ↔ Text ↔ Speech")
    parser.add_argument("--share", action="store_true", help="Create a shareable link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run server on")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--reset-models", action="store_true", help="Reset downloaded models")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.reset_models:
        # Run model reset script
        if os.name == 'nt':  # Windows
            os.system("reset_models.bat")
        else:  # Unix-like
            os.system("./reset_models.sh")
        return
    
    if args.test:
        # Run tests
        print("Running tests...")
        import unittest
        tests = unittest.defaultTestLoader.discover("tests")
        unittest.TextTestRunner().run(tests)
        return
    
    # Import here to avoid slow imports when just running tests or reset
    from src.web.app import launch_app
    
    # Launch the Gradio app
    print(f"Starting EchoLang on port {args.port}...")
    launch_app(share=args.share, server_port=args.port)

if __name__ == "__main__":
    main()