import urllib.request
import os
from pathlib import Path

def download_file(url: str, filename: str) -> None:
    path = Path("tests/fixtures/sample_docs") / filename
    if not path.exists():
        print(f"Downloading {filename}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response, open(path, "wb") as out_file:
                data = response.read()
                out_file.write(data)
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

if __name__ == "__main__":
    os.makedirs("tests/fixtures/sample_docs", exist_ok=True)
    
    download_file("https://raw.githubusercontent.com/w3c/wcag/master/LICENSE", "sample_license.md")
    download_file("https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf", "sample_pdf.pdf")
