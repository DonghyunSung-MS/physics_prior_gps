from setuptools import find_packages, setup

setup(
    name="physics prior gps",
    version="1.0",
    description="Guided Policy Search with Physics Prior",
    author="Donghyun Sung",
    author_email="dh-sung@naver.com",
    install_requires=["gym", "torch>=1.4", "osqp"],
    packages=find_packages(exclude=["docs", "tests*"]),
    keywords=["liquibase", "db migration"],
    python_requires=">=3.6",
)
