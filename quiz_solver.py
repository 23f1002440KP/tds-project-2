import json
import time
import os
import requests
import re
from urllib.parse import urljoin, urlparse
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pypdf
import pytesseract
from PIL import Image
import io
import base64
import mimetypes
from dotenv import load_dotenv

load_dotenv()

# Maximum file size (bytes) to inline as data URL when sending to the LLM.
# Default 5 MB to avoid extremely large payloads.
MAX_DATA_URL_BYTES = 5 * 1024 * 1024
from google import genai
from google.genai import types


# --- CONFIGURATION ---
# Replace with your actual LLM API details or AIPipe configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
LLM_API_KEY = os.environ.get("LLM_API_KEY")
LLM_MODEL = "google/gemini-2.5-flash" # or your specific model

class AutoSolver:
    def __init__(self, start_payload):
        self.email = start_payload['email']
        self.secret = start_payload['secret']
        self.current_url = start_payload['url']
        self.session = requests.Session()
        self.scraped_data_log = []
        
        # Setup download directory
        self.download_dir = "downloads"
        os.makedirs(self.download_dir, exist_ok=True)

    def log(self, message):
        print(f"[{time.strftime('%H:%M:%S')}] {message}")

    def download_and_extract_file(self, url, page_url):
        """Downloads files and extracts text based on type (PDF, Image, etc)"""
        try:
            full_url = urljoin(page_url, url)
            filename = os.path.basename(urlparse(full_url).path)
            filepath = os.path.join(self.download_dir, filename)
            
            self.log(f"Downloading file: {filename or full_url}")
            response = self.session.get(full_url, stream=True, timeout=60)
            # Determine filename/extension from URL or response headers
            content_type = response.headers.get('content-type', '')
            self.log(f"Download response content-type: {content_type}")

            if not filename:
                # try to choose an extension from content-type
                ct = content_type.split(';')[0] if content_type else ''
                guessed = mimetypes.guess_extension(ct) if ct else None
                if guessed and guessed.startswith('.'):
                    filename = f"downloaded_{int(time.time())}{guessed}"
                else:
                    filename = f"downloaded_{int(time.time())}.bin"
                filepath = os.path.join(self.download_dir, filename)

            # write content to disk
            total_bytes = 0
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total_bytes += len(chunk)
            self.log(f"Saved file {filepath} ({total_bytes} bytes)")

            # Extraction Logic: determine extension from filename or content-type
            if '.' in filename:
                ext = filename.lower().split('.')[-1]
            else:
                ct = content_type.split(';')[0] if content_type else ''
                guessed = mimetypes.guess_extension(ct) if ct else None
                if guessed and guessed.startswith('.'):
                    ext = guessed[1:]
                else:
                    ext = ''
            extracted_text = ""

            if ext == 'pdf':
                reader = pypdf.PdfReader(filepath)
                for page in reader.pages:
                    extracted_text += page.extract_text() + "\n"
            
            elif ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp']:
                # Image: perform OCR and also supply a data URL for the LLM
                try:
                    image = Image.open(filepath)
                    extracted_text = pytesseract.image_to_string(image)
                except Exception as e:
                    self.log(f"Image OCR failed for {filename}: {e}")
                    extracted_text = ""

                # Build data URL
                try:
                    mime, _ = mimetypes.guess_type(filename)
                    if not mime:
                        mime = f"image/{ext}"
                    with open(filepath, 'rb') as imf:
                        b64 = base64.b64encode(imf.read()).decode('ascii')
                    data_url = f"data:{mime};base64,{b64}"
                except Exception as e:
                    self.log(f"Failed to build data URL for {filename}: {e}")
                    data_url = None
                
            elif ext in ['mp3', 'wav', 'm4a', 'ogg', 'opus'] or (content_type.startswith('audio') if content_type else False):
                # Audio: attempt transcription and include transcript in returned data
                try:
                    self.log(f"Attempting transcription for {filepath}")
                    transcript = self.transcribe_audio(filepath)
                    self.log(f"Transcription result length: {len(transcript) if transcript else 0}")
                    extracted_text = transcript or ""
                except Exception as e:
                    self.log(f"Audio transcription failed for {filename}: {e}")
                    extracted_text = ""
                data_url = None
                
            result = {"filename": filename, "content_type": ext, "extracted_text": extracted_text}

            # Prefer data_url already created (images branch). Otherwise, for any non-audio
            # file types attempt to create a base64 data URL and attach it, respecting size
            # limits to prevent huge payloads.
            if ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'] and data_url:
                result['data_url'] = data_url
            else:
                # Do not inline audio files
                if not (ext in ['mp3', 'wav', 'm4a', 'ogg', 'opus'] or (content_type.startswith('audio') if content_type else False)):
                    try:
                        filesize = os.path.getsize(filepath)
                        if filesize <= MAX_DATA_URL_BYTES:
                            with open(filepath, 'rb') as ff:
                                b = ff.read()
                            mime = content_type.split(';')[0] if content_type else mimetypes.guess_type(filename)[0] or 'application/octet-stream'
                            try:
                                b64_all = base64.b64encode(b).decode('ascii')
                                result['data_url'] = f"data:{mime};base64,{b64_all}"
                            except Exception as e:
                                self.log(f"Failed to base64-encode file {filename}: {e}")
                        else:
                            self.log(f"Skipping inlining {filename} as data URL; size {filesize} > {MAX_DATA_URL_BYTES}")
                    except Exception as e:
                        self.log(f"Error while attempting to inline {filename}: {e}")

            return result

        except Exception as e:
            self.log(f"Error processing file {url}: {str(e)}")
            return None

    def transcribe_audio(self, filepath):
        """Attempt to transcribe audio file.

        Strategy:
        1. Try to use the `whisper` package if installed.
        2. Fall back to OpenAI's transcription endpoint using `LLM_API_KEY`.
        If neither is available, return an explanatory placeholder string.
        """
        # Try local whisper package
        try:
            self.log(f"transcribe_audio: trying local whisper for {filepath}")
            client = genai.Client(api_key=GEMINI_API_KEY)
    
            # Read the audio file
            with open(filepath, 'rb') as f:
                audio_bytes = f.read()
            
            # Generate transcription
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=[
                    'Generate a transcript of the speech in a way that you are giving command to someone.Without adding any extra information, just give the transcript.',
                    types.Part.from_bytes(
                        data=audio_bytes,
                        mime_type='audio/wav',  # Adjust based on your file type
                    ),
                ],
            )
            
            txt = response.text
            self.log(f"transcribe_audio: local whisper produced {len(txt)} chars")
            return txt
        except Exception as e:
            self.log(f"transcribe_audio: local whisper not available or failed: {e}")
            pass


    def is_submit_url(self, url):
        """Return True if the URL points to a '/submit' endpoint (exact segment match)."""
        try:
            parsed = urlparse(url)
            # Split path into segments and check for exact 'submit' segment
            segments = [seg.lower() for seg in parsed.path.split('/') if seg]
            return 'submit' in segments
        except Exception:
            return False

    def scrape_url(self, url):
        """Scrapes the page using Playwright to handle JS."""
        # Avoid scraping submit endpoints
        if self.is_submit_url(url):
            self.log(f"Skipping scraping of submit URL: {url}")
            return {
                "url": url,
                "text_content": "",
                "links": [],
                "files": [],
                "metadata": {"skipped": True}
            }

        self.log(f"Scraping: {url}")
        
        data_packet = {
            "url": url,
            "text_content": "",
            "html": "",
            "links": [],
            "files": [],
            "metadata": {}
        }

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True) # Set headless=False to watch it work
            page = browser.new_page()
            
            try:
                page.goto(url, wait_until="networkidle")
                
                # Get main text
                data_packet['text_content'] = page.inner_text("body")
                data_packet['metadata']['title'] = page.title()
                
                # Extract embedded images/audio (including blob URLs) by evaluating in page context.
                try:
                    media_list = page.evaluate(r"""
                    () => {
                        function arrayBufferToBase64(buffer) {
                            var binary = '';
                            var bytes = new Uint8Array(buffer);
                            var len = bytes.byteLength;
                            for (var i = 0; i < len; i++) {
                                binary += String.fromCharCode(bytes[i]);
                            }
                            return btoa(binary);
                        }

                        const nodes = [];
                        document.querySelectorAll('img, audio, source').forEach(el => {
                            const src = el.currentSrc || el.src || el.getAttribute('src');
                            if (!src) return;
                            nodes.push({tag: el.tagName.toLowerCase(), src: src});
                        });

                        return Promise.all(nodes.map(async (n) => {
                            try {
                                if (n.src.startsWith('data:')) {
                                    return {tag: n.tag, src: n.src, dataUrl: n.src};
                                } else if (n.src.startsWith('blob:')) {
                                    const resp = await fetch(n.src);
                                    const buf = await resp.arrayBuffer();
                                    const b64 = arrayBufferToBase64(buf);
                                    const ct = resp.headers.get('content-type') || '';
                                    const dataUrl = 'data:' + ct + ';base64,' + b64;
                                    return {tag: n.tag, src: n.src, dataUrl: dataUrl};
                                } else {
                                    return {tag: n.tag, src: n.src, dataUrl: null};
                                }
                            } catch (e) {
                                return {tag: n.tag, src: n.src, dataUrl: null, error: String(e)};
                            }
                        }));
                    }
                    """)
                except Exception:
                    media_list = []

                # Process embedded media found on the page
                for idx, m in enumerate(media_list or []):
                    try:
                        tag = m.get('tag')
                        src = m.get('src')
                        dataUrl = m.get('dataUrl')
                        if dataUrl:
                            # dataUrl like data:<mime>;base64,<data>
                            header, b64 = dataUrl.split(',', 1)
                            mime = header.split(':')[1].split(';')[0] if ':' in header else None
                            ext = mimetypes.guess_extension(mime) if mime else None
                            if ext and ext.startswith('.'):
                                ext = ext[1:]
                            else:
                                # fallback
                                ext = 'bin'

                            fname = f"embedded_{idx}.{ext}"
                            fpath = os.path.join(self.download_dir, fname)
                            with open(fpath, 'wb') as fh:
                                fh.write(base64.b64decode(b64))

                            if tag == 'img':
                                # OCR + data_url
                                try:
                                    image = Image.open(fpath)
                                    ocr = pytesseract.image_to_string(image)
                                except Exception:
                                    ocr = ''
                                data_packet['files'].append({
                                    'filename': fname,
                                    'content_type': ext,
                                    'extracted_text': ocr,
                                    'data_url': dataUrl,
                                    'source': src
                                })
                            elif tag == 'audio':
                                # transcribe
                                try:
                                    print(f"Transcribing embedded audio from {src}...")
                                    transcript = self.transcribe_audio(fpath)
                                except Exception:
                                    transcript = ''
                                data_packet['files'].append({
                                    'filename': fname,
                                    'content_type': ext,
                                    'extracted_text': transcript,
                                    'data_url': dataUrl,
                                    'source': src
                                })
                        else:
                            # If media has an external src (not data/blob), attempt to download it
                            if src and not src.startswith('blob:') and not src.startswith('data:'):
                                try:
                                    # download_and_extract_file expects a relative or absolute URL
                                    file_data = self.download_and_extract_file(src, url)
                                    if file_data:
                                        # mark origin as media tag
                                        file_data['source'] = src
                                        file_data['tag'] = tag
                                        data_packet['files'].append(file_data)
                                except Exception as e:
                                    self.log(f"Failed to download media from {src}: {e}")
                    except Exception as e:
                        self.log(f"Error processing embedded media: {e}")

                # Get HTML for link parsing
                html = page.content()
                data_packet['html'] = html
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find all links
                anchors = soup.find_all('a', href=True)
                
                # Separate Navigation links from File links
                file_extensions = ['.pdf', '.csv', '.png', '.jpg', '.mp3', '.wav', '.json','.opus','.jpeg','.gif','.bmp','.webp','.m4a','.ogg']
                
                for a in anchors:
                    href = a['href']
                    full_link = urljoin(url, href)

                    # Skip any links that point to submit endpoints
                    if self.is_submit_url(full_link):
                        self.log(f"Skipping link to submit URL: {full_link}")
                        continue

                    if any(href.lower().endswith(ext) for ext in file_extensions):
                        # It is a file, download and extract
                        file_data = self.download_and_extract_file(href, url)
                        if file_data:
                            data_packet['files'].append(file_data)
                    else:
                        # It is a webpage link
                        data_packet['links'].append(full_link)

            except Exception as e:
                self.log(f"Scraping error: {e}")
            finally:
                browser.close()
                
        return data_packet

    def get_answer_from_llm(self, json_data, feedback=None):
        """
        Passes the scraped JSON to the LLM (simulating aipipe).
        If `feedback` is provided (a dict with keys like 'previous_answer' and
        'submission_reason'), that feedback will be included in the prompt so the
        model can revise its answer.

        Retries up to 5 times within 3 minutes for transient LLM/API errors.
        """

        system_prompt = """
          You are an intelligent agent designed to solve Capture The Flag (CTF) and Data Analysis quizzes.
          1. Analyze the provided JSON data, which represents a scraped website.
          2. Use the following fields (priority order) to find and answer the question:
              - `page_source`: the full HTML source of the page (preferred for precise extraction);
              - `files`: an array of downloaded/extracted files. Each file may include:
                 * `filename`, `content_type`, `extracted_text` (OCR or text extraction),
                 * `data_url` (a base64 data: URL for non-audio files, when available),
                 * for audio files, `extracted_text` will contain a transcript when available.
              - `text_content`: the visible rendered text of the page.
          3. Do NOT attempt to fetch external URLs yourself. Use the provided `data_url` attachments and
              provided transcripts instead of network access.
          4. If submission feedback is provided, use it to revise your answer and briefly explain why you changed it.
          5. If calculations or code are required, perform them and provide only the final answer.
          6. OUTPUT ONLY THE FINAL ANSWER. Do not include any surrounding commentary or explanatory text.
        """

        # Convert JSON to string prompt
        user_prompt = json.dumps(json_data, indent=2)
        print(user_prompt)
        if feedback:
            # Append feedback as an additional user message instead of mutating the main prompt
            feedback_text = json.dumps(feedback, indent=2)
        else:
            feedback_text = None

        start_time = time.time()
        attempts = 0
        max_attempts = 5

        while attempts < max_attempts and (time.time() - start_time) < 180:
            try:
                self.log(f"Asking LLM (Attempt {attempts + 1})...")

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]

                if feedback_text:
                    messages.append({"role": "user", "content": f"Submission feedback (use to revise answer):\n{feedback_text}"})

                response = requests.post(
                    "https://aipipe.org/openrouter/v1/chat/completions",
                    headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                    json={
                        "model": LLM_MODEL,
                        "messages": messages
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    answer = response.json()['choices'][0]['message']['content'].strip()
                    print(answer)
                    # Clean up answer (remove markdown fences if any)
                    answer = answer.replace("```", "").strip()
                    self.log(f"LLM Answer: {answer}")
                    return answer
                else:
                    self.log(f"LLM Error: {response.text}")

            except Exception as e:
                self.log(f"LLM Exception: {e}")

            attempts += 1
            time.sleep(5) # Wait before retry

        return None

    def submit_answer(self, answer, quiz_url):
        """Submits the answer to the submit endpoint."""
        
        # Heuristic: usually submit URL is scrape URL domain + /submit, or hardcoded
        # The prompt says use specific URL if not found, let's try the dynamic one first
        submit_url = "https://tds-llm-analysis.s-anand.net/submit"
        
        payload = {
            "email": self.email,
            "secret": self.secret,
            "url": quiz_url,
            "answer": answer
        }
        
        self.log(f"Submitting payload: {json.dumps(payload)}")
        
        try:
            response = requests.post(submit_url, json=payload)
            self.log(f"Submission Response: {response.text}")
            return response.json()
        except Exception as e:
            self.log(f"Submission Failed: {e}")
            return None

    def run(self):
        """Main Loop"""
        while self.current_url:
            self.log(f"--- Processing Level: {self.current_url} ---")
            
            # 1. Scrape Main Page
            scraped_data = self.scrape_url(self.current_url)
            
            # 2. Check for Linked Pages (Recursive Scrape Depth 1)
            # If the instructions are hidden in linked pages, we grab them.
            # We limit this to avoid scraping the whole internet.
            base_domain = urlparse(self.current_url).netloc
            for link in scraped_data['links']:
                if urlparse(link).netloc == base_domain:
                    # Skip any submit endpoints even if internal
                    if self.is_submit_url(link):
                        self.log(f"Skipping follow of submit URL: {link}")
                        continue
                    # Only scrape internal links to append to context
                    sub_data = self.scrape_url(link)
                    scraped_data[f"sub_page_{link}"] = sub_data

            # 3. Save Scrape Data
            self.scraped_data_log.append(scraped_data)
            with open('scraped_data.json', 'w') as f:
                json.dump(self.scraped_data_log, f, indent=2)

            # 4. Get Answer from LLM
            # print(scraped_data)
            answer = self.get_answer_from_llm(scraped_data)
            
            if not answer:
                self.log("Failed to get answer after retries.")
                # Try to continue to a discovered link on the page instead of stopping.
                next_url = None
                for l in scraped_data.get('links', []) or []:
                    if not self.is_submit_url(l):
                        next_url = l
                        break

                if next_url:
                    self.current_url = next_url
                    self.log("No LLM answer — moving to next discovered URL to continue.")
                    continue
                else:
                    self.log("No answer and no next URL to proceed. Stopping.")
                    break

            # 5. Submit
            result = self.submit_answer(answer, self.current_url)

            # 6. Check Result & Loop
            if result and result.get('correct', False):
                self.log("Answer Correct!")
                next_url = result.get('url')
                if next_url:
                    self.current_url = next_url
                    self.log("Moving to next level...")
                else:
                    self.log("No new URL provided. Challenge Complete.")
                    break
            else:
                # If server provided a reason for rejection, send it back to the LLM
                reason = None
                if isinstance(result, dict):
                    reason = result.get('reason')

                if reason:
                    self.log(f"Answer Incorrect. Server reason: {reason}. Requesting revised answer from LLM.")
                    prev_answer = answer
                    max_revision_attempts = 3
                    rev_attempts = 0

                    # Keep track of the last submission response so we can look for a
                    # server-provided next `url` even if the answer never becomes correct.
                    last_submission = result if isinstance(result, dict) else None
                    result2 = None

                    while rev_attempts < max_revision_attempts:
                        feedback = {"previous_answer": prev_answer, "submission_reason": reason}
                        revised = self.get_answer_from_llm(scraped_data, feedback=feedback)
                        if not revised:
                            self.log("LLM failed to provide a revised answer.")
                            break

                        self.log(f"Submitting revised answer (attempt {rev_attempts + 1}): {revised}")
                        result2 = self.submit_answer(revised, self.current_url)

                        # update last submission snapshot
                        if isinstance(result2, dict):
                            last_submission = result2

                        if result2 and result2.get('correct', False):
                            self.log("Revised answer correct!")
                            next_url = result2.get('url')
                            if next_url:
                                self.current_url = next_url
                                self.log("Moving to next level...")
                            else:
                                self.log("No new URL provided. Challenge Complete.")
                                self.current_url = None
                            break

                        # Not correct: update reason and previous answer and retry
                        prev_answer = revised
                        if isinstance(result2, dict):
                            reason = result2.get('reason')
                        else:
                            reason = None

                        rev_attempts += 1
                        self.log(f"Revised attempt {rev_attempts} incorrect. Server reason: {reason}")

                        if not reason:
                            # No further guidance from server, stop trying
                            break

                    # If we exited the loop without a correct answer, attempt to continue
                    # to the `url` provided by the server (if any) instead of stopping.
                    if not (result2 and isinstance(result2, dict) and result2.get('correct', False)):
                        next_url = None
                        if isinstance(last_submission, dict):
                            next_url = last_submission.get('url')

                        if next_url:
                            self.current_url = next_url
                            self.log("Exceeded revision attempts — moving to server-provided next URL.")
                            continue
                        else:
                            self.log("Exceeded revision attempts and no next URL provided. Stopping.")
                            break
                else:
                    # If the server did not provide a reason, try to continue if the server
                    # returned a next `url` or if the page had other internal links to follow.
                    next_url = None
                    if isinstance(result, dict):
                        next_url = result.get('url')

                    if not next_url:
                        for l in scraped_data.get('links', []) or []:
                            if not self.is_submit_url(l):
                                next_url = l
                                break

                    if next_url:
                        self.current_url = next_url
                        self.log("Answer incorrect but moving to next URL to continue.")
                        continue
                    else:
                        self.log("Answer Incorrect and no reason provided. Stopping.")
                        break

# --- EXECUTION ---
if __name__ == "__main__":
    # 1. Receive the initial POST request (Simulated here)
    initial_payload = {
        "email": "23f1002440@ds.study.iitm.ac.in",
        "secret": "ItsaSecret123",
        "url": "https://tds-llm-analysis.s-anand.net/demo"
    }
    
    bot = AutoSolver(initial_payload)
    bot.run()