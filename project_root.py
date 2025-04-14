from pathlib import Path

PROJECT_ROOT = Path(__file__).parent  # This file's directory = project root

def get_path(*parts):
    """Safe path builder: get_path('data', 'audio') -> PROJECT_ROOT/data/audio"""
    return PROJECT_ROOT.joinpath(*parts).resolve()