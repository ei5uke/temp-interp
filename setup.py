from setuptools import find_packages, setup

def get_requirements(path: str):
    # return [l.strip() for l in open(path)]
    with open(path) as f:
        lines = f.read().splitlines()
    reqs = []
    for line in lines:
        line = line.strip()
        # Skip blank lines, comments, and pip flags
        if not line or line.startswith(("#", "-", "--")):
            continue
        reqs.append(line)
    return reqs

setup(
    name="temp_interp",
    version="0.0.1",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)