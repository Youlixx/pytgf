import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytgl",
    version="1.0.0",
    author="Theo Combey",
    author_email="combey.theo@hotmail.com",
    description="A simple tile engine based on OpenGL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Youlixx/pytgl",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'moderngl',
        'pyglet'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
