from setuptools import setup, find_packages

setup(
    name="sonnet-prose-eval",
    version="0.1.0",
    packages=find_packages(),
    py_modules=["eval_runner", "verify_eval_suite"],
    install_requires=[
        "google-cloud-aiplatform",
        "google-cloud-firestore",
        "requests",
        "certifi",
    ],
)
