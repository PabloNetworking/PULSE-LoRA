import setuptools

with open("README.md", "r", encoding="utf-8") as handle:
    long_description = handle.read()

setuptools.setup(
    name="pulse-lora",
    version="0.1.0.dev0",
    description="PULSE-LoRA: decentralized LoRA fine-tuning under heterogeneous compute and "
                "dynamic peer interactions (work in progress)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=[package for package in setuptools.find_packages()
              if package.startswith("federatedscope")],
    python_requires=">=3.9",
    license="Apache License 2.0",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
