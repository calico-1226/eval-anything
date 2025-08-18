from setuptools import setup, find_packages


setup(
    name="flag_safety",
    version="0.0.0",
    description="FlagSafety",
    packages=find_packages(include=["flag_safety", "flag_safety.*"]),
    include_package_data=True,
    install_requires=[],
)
