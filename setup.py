from setuptools import setup, find_packages

setup(
    name='vitm',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0',  # Let users pick their CUDA variant (avoid hardcoding '+cu*')
        'torchvision>=0.15',
        'torchaudio>=2.0',
        'numpy>=1.21',
        'pandas>=1.5',
        'matplotlib>=3.5',
        'opencv-python>=4.5',
        'absl-py>=1.0.0',
        'fvcore>=0.1.5',
        'iopath>=0.1.9',
        'hydra-core>=1.3.0',
        'omegaconf>=2.2.0',
        'tqdm>=4.64',
        'pyyaml>=6.0',
        'rich>=13.0',
        'tabulate>=0.8',
        'setuptools',
        'wheel',
        'twine',
        'mkdocs',
        'mkdocs-material',
        'mkdocstrings[python]',
    ],
)
