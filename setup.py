from setuptools import setup, find_packages


setup(
    name="eval_anything",
    version="0.0.0",
    description="FlagSafety",
    packages=find_packages(include=["eval_anything", "eval_anything.*"]),
    include_package_data=True,
    install_requires=[],
)
