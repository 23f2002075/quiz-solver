#!/bin/bash
set -o errexit

pip install --upgrade pip
pip install -r requirements.txt

# Install system dependencies for Tesseract
apt-get update && apt-get install -y tesseract-ocr

# Install Chrome for Selenium
apt-get install -y chromium-browser chromium-chromedriver
