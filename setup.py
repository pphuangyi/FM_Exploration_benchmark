from setuptools import setup

setup(
    name = "sPHENIX_Benchmarks",
    version = "0.0.1.dev",
    author = "TBD",
    author_email = "TBD",
    description = ("Benchmark models for sPHENIX data"),
    keywords = "foundation model, high-energy physics simulation",
    license = "BSD 3-Clause 'New' or 'Revised' License",
    # url = "https://github.com/BNL-DAQ-LDRD/NeuralCompression",
    packages=['sphenix_benchmark',],
    long_description="",
    install_requires=[
        "torch",
        "torch_geometric",
        "numpy",
        "pandas",
        "matplotlib",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: BSD 3-Clause 'New' or 'Revised' License",
    ],
)
