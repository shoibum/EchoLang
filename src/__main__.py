"""
Main entry point for the multilingual STT/TTS application.
"""

import argparse
from src.web.app import create_and_launch_app

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Multilingual Speech-to-Text and Text-to-Speech Application"
    )
    parser.add_argument(
        "--share", 
        action="store_true", 
        help="Create a shareable link"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=7860, 
        help="Port to run the application on"
    )
    
    args = parser.parse_args()
    
    # Launch the application
    create_and_launch_app(share=args.share, server_port=args.port)

if __name__ == "__main__":
    main()