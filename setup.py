"""Setup configuration for the conversation_vad_labeler package."""

from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).parent
README = (HERE / "readme.md").read_text(encoding="utf-8")


setup(
    name="conversation_vad_labeler",
    version="0.1.0",
    description="Conversation VAD labeling and transcription pipeline",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Hanlu He",
    author_email="hahea@dtu.dk",
    packages=find_packages(include=["conversation_vad_labeler", "conversation_vad_labeler.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.23",
        "pandas>=1.5",
        "scipy>=1.9",
        "soundfile>=0.12",
        "torch>=2.0",
        "transformers>=4.38",
        "tqdm>=4.65",
    ],
)
