from setuptools import find_packages, setup

#with open('requirements.txt') as f:
#    required = f.read().splitlines()

install_requires = [
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "opencv-python",
    "pillow",
    "argparse",
    "pyyaml",
    "scikit-learn",
    "wandb",
    "uuid",
    "pytorch-lightning",
    "pytorch-lightning-bolts",
    "ray[tune]",
    "boto3",
    "botocore",
    "sagemaker",
    "python-dotenv",
]

setup(
    name='deeppype',
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.9",
    version='0.1.1',
    description='The AWS based Deep Learning Pipeline Framework',
    author='Marco Placenti',
    author_email="s202798@student.dtu.dk",
    license='MIT',
    project_urls={
        "Source Code": 'https://github.com/marcoplacenti/deeppype'
    }
)
