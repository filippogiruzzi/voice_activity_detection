"""Setup file to install VAD as a package."""
from setuptools import find_packages, setup

setup(
    name="voice_activity_detection",
    version="0.0",
    description="Voice Activity Detection project",
    author="Filippo Giruzzi",
    author_email="filippo.giruzzi@gmail.com",
    packages=find_packages(),
    zip_safe=False,
)
