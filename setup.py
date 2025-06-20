#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="trading_system",
    version="1.0.0",
    description="Time-MoE Trading System - Advanced algorithmic trading with GPU acceleration",
    author="Time-MoE Trading",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch==2.6.0",
        "transformers==4.52.4", 
        "numpy==1.21.5",
        "scipy==1.8.0",
        "aiohttp==3.12.13",
        "python-dotenv==1.1.0",
        "pytz==2022.1",
        "alpaca-trade-api==3.2.0",
        "polygon-api-client==1.14.6",
        "websocket-client==1.2.3",
        "psutil==5.9.0",
        "flash-attn==2.8.0.post2",
        "requests==2.25.1",
    ],
    entry_points={
        'console_scripts': [
            'start-trading=trading_system.start_trading:main',
            'test-trading=trading_system.test_system:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)