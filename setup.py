from setuptools import setup, find_packages

setup(
    name="driver-style-classifier",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="ML system for driver style classification and ECU optimization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.13.0",
        "scikit-learn>=1.3.0", 
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "joblib>=1.3.0",
        "python-dateutil>=2.8.0"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
