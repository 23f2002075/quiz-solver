from langchain_core.tools import tool
import requests
from bs4 import BeautifulSoup
import subprocess
import os
import json
import tempfile

# -------------------------------------------------
# TOOLS
# -------------------------------------------------

@tool
def get_rendered_html(url: str) -> str:
    """Fetch and render HTML content from a URL using Selenium for JavaScript execution.
    
    This tool fully renders JavaScript and waits for dynamic content to load.
    Returns complete HTML source with all rendered elements.
    """
    try:
        print(f"\n[HTML] Fetching: {url}")
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        import time
        
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-web-resources")
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(15)
        driver.get(url)
        
        # Wait for body to load
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
        except:
            pass  # Continue anyway
        
        # Wait for common UI elements
        time.sleep(1)
        
        html_content = driver.page_source
        
        # Extract visible text for context
        visible_text_count = len(driver.find_element(By.TAG_NAME, "body").text)
        
        driver.quit()
        
        print(f"[HTML] Success: {len(html_content)} bytes HTML, ~{visible_text_count} visible chars")
        return html_content
    except Exception as e:
        error_msg = f"HTML fetch failed: {str(e)}"
        print(f"[HTML] Error: {error_msg}")
        return error_msg


@tool
def download_file(url: str, filepath: str) -> str:
    """Download a file from a URL and save it locally.
    
    Returns file info: size, type, and location.
    """
    try:
        print(f"\n[DOWNLOAD] Fetching from: {url}")
        response = requests.get(url, timeout=30, allow_redirects=True)
        response.raise_for_status()
        
        file_size = len(response.content)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        # Get file info
        file_ext = filepath.split('.')[-1].lower() if '.' in filepath else 'unknown'
        
        print(f"[DOWNLOAD] Success: {filepath} ({file_size} bytes, type: {file_ext})")
        return f"File downloaded: {filepath}\nSize: {file_size} bytes\nType: {file_ext}"
    except requests.exceptions.Timeout:
        error_msg = f"Download timeout for {url}"
        print(f"[DOWNLOAD] Error: {error_msg}")
        return error_msg
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP error {e.response.status_code} downloading {url}"
        print(f"[DOWNLOAD] Error: {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"Download failed: {str(e)}"
        print(f"[DOWNLOAD] Error: {error_msg}")
        return error_msg


@tool
def post_request(url: str, json_data: dict, headers: dict | None = None) -> str:
    """Send a POST request with JSON data and analyze response.
    
    Returns: Full response with status, headers, and parsed content.
    """
    try:
        if headers is None:
            headers = {'Content-Type': 'application/json'}
        
        print(f"\n[POST] Submitting to: {url}")
        print(f"[POST] Payload: {json.dumps(json_data, indent=2)[:200]}...")
        
        response = requests.post(
            url,
            json=json_data,
            headers=headers,
            timeout=30,
            allow_redirects=True
        )
        
        print(f"[POST] Status: {response.status_code}")
        
        # Parse response
        result_data = {
            "status_code": response.status_code,
            "success": 200 <= response.status_code < 300,
        }
        
        # Try JSON parsing
        try:
            parsed_json = response.json()
            result_data["body"] = parsed_json
            result_data["format"] = "json"
            print(f"[POST] Response (JSON): {json.dumps(parsed_json, indent=2)[:300]}...")
        except:
            # Try text
            result_data["body"] = response.text[:500]
            result_data["format"] = "text"
            print(f"[POST] Response (text): {response.text[:100]}...")
        
        return json.dumps(result_data, indent=2)
        
    except requests.exceptions.Timeout:
        error_data = {"status": "timeout", "message": "Request exceeded 30s timeout"}
        print(f"[POST] Error: Timeout")
        return json.dumps(error_data)
    except requests.exceptions.ConnectionError as e:
        error_data = {"status": "connection_error", "message": str(e)}
        print(f"[POST] Error: Connection failed")
        return json.dumps(error_data)
    except Exception as e:
        error_data = {"status": "error", "message": str(e)}
        print(f"[POST] Error: {str(e)}")
        return json.dumps(error_data)


@tool
def run_code(code: str, language: str = "python") -> str:
    """Execute code locally with comprehensive output.
    
    Returns: Execution output, errors, and status.
    60s timeout for Python execution.
    """
    try:
        if language.lower() != "python":
            return f"Language '{language}' not supported. Use 'python' only."
        
        print(f"\n[EXEC] Running Python code ({len(code)} chars)")
        
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=os.getcwd()
        )
        
        output = result.stdout.strip()
        errors = result.stderr.strip()
        
        # Build comprehensive output
        exec_result = {
            "status": "success" if result.returncode == 0 else "error",
            "return_code": result.returncode,
            "output": output if output else "(no output)",
            "errors": errors if errors else None
        }
        
        if output:
            print(f"[EXEC] Output: {output[:100]}")
        if errors:
            print(f"[EXEC] Errors: {errors[:100]}")
        if result.returncode != 0:
            print(f"[EXEC] Failed with code {result.returncode}")
        
        return json.dumps(exec_result, indent=2)
        
    except subprocess.TimeoutExpired:
        timeout_msg = "Code execution exceeded 60s timeout"
        print(f"[EXEC] {timeout_msg}")
        return json.dumps({"status": "timeout", "message": timeout_msg})
    except Exception as e:
        error_msg = f"Execution failed: {str(e)}"
        print(f"[EXEC] Error: {error_msg}")
        return json.dumps({"status": "error", "message": error_msg})


@tool
def add_dependencies(packages: list[str]) -> str:
    """Install Python packages using pip with detailed status.
    
    Args:
        packages: List of package names to install
        
    Returns: Installation summary with success/failure details
    """
    try:
        print(f"\n[PIP] Installing {len(packages)} packages: {', '.join(packages)}")
        
        installed = []
        failed = []
        already_present = []
        
        for package in packages:
            try:
                # Check if already installed
                result = subprocess.run(
                    ["python", "-c", f"import {package.split('[')[0].replace('-', '_')}"],
                    capture_output=True,
                    timeout=10
                )
                if result.returncode == 0:
                    already_present.append(package)
                    print(f"[PIP] Already present: {package}")
                    continue
            except:
                pass
            
            # Try to install
            result = subprocess.run(
                ["pip", "install", package, "-q"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                installed.append(package)
                print(f"[PIP] Installed: {package}")
            else:
                failed.append(package)
                print(f"[PIP] Failed: {package} - {result.stderr[:100] if result.stderr else ''}")
        
        summary = {
            "installed": installed,
            "already_present": already_present,
            "failed": failed,
            "total_count": len(packages),
            "success_count": len(installed) + len(already_present)
        }
        
        return json.dumps(summary, indent=2)
        
    except Exception as e:
        error_msg = f"Package installation error: {str(e)}"
        print(f"[PIP] Error: {error_msg}")
        return json.dumps({"status": "error", "message": error_msg})


# -------------------------------------------------
# SPECIALIZED PROCESSING TOOLS
# -------------------------------------------------

@tool
def extract_pdf_text(filepath: str) -> str:
    """Extract text from PDF with detailed analysis.
    
    Tries pdfplumber first, falls back to PyPDF2.
    Returns: Full text + page count + character count.
    """
    try:
        print(f"\n[PDF] Extracting: {filepath}")
        text = ""
        page_count = 0
        
        try:
            import pdfplumber
            with pdfplumber.open(filepath) as pdf:
                page_count = len(pdf.pages)
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    text += f"\n--- Page {i+1} ---\n{page_text}"
            print(f"[PDF] Success (pdfplumber): {page_count} pages, {len(text)} chars")
            
        except ImportError:
            from PyPDF2 import PdfReader
            with open(filepath, 'rb') as f:
                reader = PdfReader(f)
                page_count = len(reader.pages)
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    text += f"\n--- Page {i+1} ---\n{page_text}"
            print(f"[PDF] Success (PyPDF2): {page_count} pages, {len(text)} chars")
        
        summary = {
            "format": "pdf",
            "pages": page_count,
            "characters": len(text),
            "content": text[:500] + "..." if len(text) > 500 else text,
            "full_text": text
        }
        return json.dumps(summary, indent=2)
        
    except Exception as e:
        error_msg = f"PDF extraction failed: {str(e)}"
        print(f"[PDF] Error: {error_msg}")
        return json.dumps({"status": "error", "message": error_msg})


@tool
def extract_audio_text(filepath: str) -> str:
    """Extract text from audio using speech recognition.
    
    Tries Google speech recognition, inspects audio if recognition fails.
    Returns: Transcribed text + audio details.
    """
    try:
        print(f"\n[AUDIO] Processing: {filepath}")
        
        # First try speech recognition
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with sr.AudioFile(filepath) as source:
                audio = recognizer.record(source)
            
            text = recognizer.recognize_google(audio)
            print(f"[AUDIO] Transcribed: {text[:100]}")
            
            return json.dumps({
                "format": "audio",
                "method": "speech_recognition",
                "text": text,
                "status": "success"
            }, indent=2)
            
        except Exception as sr_error:
            print(f"[AUDIO] Speech recognition failed: {str(sr_error)[:50]}")
            
            # Fallback: analyze audio metadata
            try:
                import librosa
                y, sr_val = librosa.load(filepath)
                duration = librosa.get_duration(y=y, sr=sr_val)
                
                # Get audio statistics
                rms = librosa.feature.rms(y=y)[0]
                avg_energy = float(rms.mean())
                
                print(f"[AUDIO] Audio info: {len(y)} samples, {sr_val}Hz, {duration:.2f}s")
                
                return json.dumps({
                    "format": "audio",
                    "method": "audio_analysis",
                    "samples": len(y),
                    "sample_rate": sr_val,
                    "duration_seconds": float(duration),
                    "average_energy": float(avg_energy),
                    "note": "Could not transcribe - audio analysis provided"
                }, indent=2)
                
            except Exception as librosa_error:
                return json.dumps({
                    "format": "audio",
                    "status": "error",
                    "message": f"Both transcription and analysis failed: {str(librosa_error)}"
                }, indent=2)
                
    except Exception as e:
        error_msg = f"Audio processing failed: {str(e)}"
        print(f"[AUDIO] Error: {error_msg}")
        return json.dumps({"status": "error", "message": error_msg})


@tool
def extract_parquet_data(filepath: str) -> str:
    """Extract and analyze parquet file with detailed statistics.
    
    Returns: Schema, shape, dtypes, and first rows.
    """
    try:
        print(f"\n[PARQUET] Reading: {filepath}")
        
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(filepath)
            df = table.to_pandas()
        except ImportError:
            import fastparquet
            pf = fastparquet.ParquetFile(filepath)
            df = pf.to_pandas()
        
        # Build comprehensive analysis
        analysis = {
            "format": "parquet",
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "shape": f"{len(df)} rows × {len(df.columns)} cols",
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
            "preview": df.head(10).to_dict('records'),
            "statistics": {
                col: {
                    "type": str(df[col].dtype),
                    "null_count": int(df[col].isna().sum()),
                    "null_percent": float(df[col].isna().sum() / len(df) * 100)
                }
                for col in df.columns
            }
        }
        
        print(f"[PARQUET] Success: {analysis['rows']} rows × {analysis['columns']} cols")
        return json.dumps(analysis, indent=2)
        
    except Exception as e:
        error_msg = f"Parquet reading failed: {str(e)}"
        print(f"[PARQUET] Error: {error_msg}")
        return json.dumps({"status": "error", "message": error_msg})


@tool
def extract_image_text_ocr(filepath: str) -> str:
    """Extract text from image using Tesseract OCR.
    
    Returns: Extracted text + image metadata + confidence.
    """
    try:
        print(f"\n[OCR] Processing: {filepath}")
        from PIL import Image
        import pytesseract
        
        img = Image.open(filepath)
        
        # Get image metadata
        img_format = img.format
        img_size = img.size
        img_mode = img.mode
        
        # Extract text with config for better accuracy
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(img, config=custom_config)
        
        # Get detailed data
        data = pytesseract.image_to_data(img, output_type='dict')
        
        # Calculate confidence (average of non-zero confidences)
        confidences = [int(c) for c in data.get('confidence', []) if int(c) > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        analysis = {
            "format": "image_ocr",
            "image_format": img_format,
            "image_size": img_size,
            "image_mode": img_mode,
            "text": text.strip() if text.strip() else "(no text detected)",
            "text_length": len(text.strip()),
            "confidence": float(avg_confidence),
            "words_detected": len([c for c in confidences if c > 30]),
            "status": "success" if text.strip() else "no_text"
        }
        
        print(f"[OCR] Success: {len(text)} chars, {float(avg_confidence):.1f}% confidence")
        return json.dumps(analysis, indent=2)
        
    except Exception as e:
        error_msg = f"OCR extraction failed: {str(e)}"
        print(f"[OCR] Error: {error_msg}")
        return json.dumps({"status": "error", "message": error_msg})


@tool
def extract_steganography(filepath: str) -> str:
    """Extract hidden message from steganographic image.
    
    Supports: LSB (Least Significant Bit) steganography.
    Returns: Hidden message + extraction details.
    """
    try:
        print(f"\n[STEG] Extracting: {filepath}")
        from PIL import Image
        import numpy as np
        
        img = Image.open(filepath)
        pixels = np.array(img)
        
        # Extract LSB (Least Significant Bit) steganography
        binary_data = ""
        
        if len(pixels.shape) == 3:  # RGB image
            for y in range(pixels.shape[0]):
                for x in range(pixels.shape[1]):
                    pixel = pixels[y, x]
                    # Get LSB from first channel (R)
                    binary_data += str(int(pixel[0]) & 1)
        else:  # Grayscale
            for y in range(pixels.shape[0]):
                for x in range(pixels.shape[1]):
                    pixel = int(pixels[y, x])
                    binary_data += str(pixel & 1)
        
        # Convert binary to text
        message = ""
        for i in range(0, len(binary_data) - 7, 8):
            byte = binary_data[i:i+8]
            try:
                char_code = int(byte, 2)
                if 32 <= char_code <= 126:  # Printable ASCII
                    message += chr(char_code)
                elif char_code == 0:  # Null terminator - end of message
                    break
                else:
                    break
            except:
                break
        
        result = message.strip()
        
        analysis = {
            "format": "image_steg",
            "image_size": img.size,
            "image_mode": img.mode,
            "total_pixels": pixels.size if len(pixels.shape) == 2 else pixels.shape[0] * pixels.shape[1],
            "bits_extracted": len(binary_data),
            "message": result if result else "(no hidden message)",
            "message_length": len(result),
            "status": "found" if result else "not_found"
        }
        
        print(f"[STEG] {'Found' if result else 'Not found'}: {result[:50] if result else 'no message'}")
        return json.dumps(analysis, indent=2)
        
    except Exception as e:
        error_msg = f"Steganography extraction failed: {str(e)}"
        print(f"[STEG] Error: {error_msg}")
        return json.dumps({"status": "error", "message": error_msg})
