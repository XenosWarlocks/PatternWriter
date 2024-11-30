# On google colab, run this line first:
# !pip install rich tqdm beautifulsoup4 nameparser email-validator pandas requests

# updated
import os
import re
import sys
import time
import random
import logging
import requests
import sqlite3
import pandas as pd

from tqdm import tqdm
from rich.progress import Progress
from bs4 import BeautifulSoup
from nameparser import HumanName
from urllib.parse import urlparse
from email_validator import validate_email, EmailNotValidError
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from rich.console import Console
from rich.prompt import Prompt
from google.colab import files

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("email_verifier.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

class EnhancedEmailVerifier:
    # Class constants
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:86.0) Gecko/20100101 Firefox/86.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    ]

    DEFAULT_PATTERNS = [
        '{first}.{last}', '{first}_{last}', '{first}{last}', '{f}{last}',
        '{first}.{l}', '{f}.{l}', '{f}.{last}', '{f}{l}', '{first}', '{last}'
    ]

    def __init__(self, db_path='visited_urls.db', max_retries=3):
        """
        Initialize the Enhanced Email Verifier with advanced configurations.

        :param db_path: Path to the SQLite database for caching
        :param max_retries: Maximum number of retries for web requests
        """
        self.db_path = db_path
        self.max_retries = max_retries
        self.error_log_file = "enhanced_email_validation_errors.txt"
        # Setup logging and error handling
        self.console = Console()
        self.create_errors_log_file_if_not_exists()

        # URL and request management
        self.visited_urls = self.load_cached_urls()
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'DNT': '1'
        })

        # Configuration flags
        self.validate_emails = self.ask_for_email_validation()

    def save_cached_urls(self):
        """
        Save visited URLs to SQLite database.
        Added to fix the __del__ method error.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("DELETE FROM visited_urls")  # Clear old entries
            for url in self.visited_urls:
                conn.execute("INSERT OR REPLACE INTO visited_urls (url) VALUES (?)", (url,))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logging.error(f"Error saving cached URLs: {e}")

    def load_cached_urls(self):
        """
        Load and manage visited URLs with SQLite.
        Implements a more robust URL caching mechanism.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS visited_urls (
                    url TEXT PRIMARY KEY,
                    last_visited TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Optional: Clean up old entries (e.g., older than 30 days)
            conn.execute('DELETE FROM visited_urls WHERE last_visited < datetime("now", "-30 days")')

            urls = {row[0] for row in conn.execute("SELECT url FROM visited_urls")}
            conn.close()
            return urls
        except sqlite3.Error as e:
            logging.error(f"Database initialization error: {e}")
            return set()

    def create_errors_log_file_if_not_exists(self):
        """Create error log file if it doesn't exist."""
        if not os.path.exists(self.error_log_file):
            with open(self.error_log_file, 'w') as log_file:
                log_file.write("Enhanced Email Verification Error Log\n")
                log_file.write("=" * 50 + "\n")

    def ask_for_email_validation(self):
        """
        Interactive email validation preference with rich console and arrow key navigation.
        """
        try:
            # Use rich's Prompt for better input handling
            options = ["Yes", "No"]
            choice = Prompt.ask(
                "[cyan]Would you like to validate email addresses?[/cyan]",
                choices=options,
                default="Yes"
            )
            return choice.lower() == "yes"
        except Exception as e:
            logging.error(f"Error in email validation preference: {e}")
            return False


    def generate_emails(self, first_name, last_name, company_url, pattern=None):
        """
        Advanced email generation with multiple fallback patterns.

        :param first_name: First name
        :param last_name: Last name
        :param company_url: Company website URL
        :param pattern: Optional specific email pattern
        :return: Generated email or None
        """
        try:
            if not (first_name and last_name and company_url):
                return None

            # Normalize URL
            company_url = company_url if company_url.startswith(('http://', 'https://')) else f'https://{company_url}'
            domain = urlparse(company_url).netloc.replace('www.', '')

            # Use provided pattern or select from defaults
            patterns = [pattern] if pattern else self.DEFAULT_PATTERNS

            placeholders = {
                '{first}': first_name.lower(),
                '{f}': first_name[0].lower(),
                '{last}': last_name.lower(),
                '{l}': last_name[0].lower(),
            }

            # Try multiple patterns until a valid email is generated
            for pattern in patterns:
                try:
                    email = pattern
                    for placeholder, value in placeholders.items():
                        email = email.replace(placeholder, value)

                    full_email = f"{email}@{domain}"
                    return full_email
                except Exception as pattern_error:
                    logging.warning(f"Pattern generation error: {pattern_error}")

            return None

        except Exception as e:
            logging.error(f"Email generation error: {e}")
            return None


    def verify_email(self, email):
        """
        Enhanced email verification with corrected validation parameters.

        :param email: Email address to verify
        :return: Verification result (True/False)
        """
        if not email or not self.validate_emails:
            return False

        try:
            # Updated validation parameters
            validated_email = validate_email(
                email,
                check_deliverability=True,
                allow_smtputf8=False,
                timeout=10
            )
            return True
        except EmailNotValidError as e:
            # Log specific validation errors
            self.log_error("EmailValidationError", f"Invalid email {email}: {str(e)}")
            return False
        except Exception as e:
            # Log unexpected errors
            self.log_error("EmailVerificationError", f"Verification error for {email}: {str(e)}")
            return False

    def extract_phone_number(self, company_url):
        """
        Advanced phone number extraction with skip mechanism for problematic URLs.

        :param company_url: Company website URL
        :return: Extracted phone number or None
        phone_patterns = [
                    r'(?:\+?1[-\s.]?)?(?:\(\d{3}\)|\d{3})[-\s.]?\d{3}[-\s.]?\d{4}',  # US format
                    r'\+?(\d{1,3})?[-. ]?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}',  # International format
                    r'(?:tel|phone|telephone)[:]\s*(\+?1?\s*\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4})',  # With context
                ]
        """
        if not company_url:
            return None

        try:

          # Normalize URL
          company_url = company_url if company_url.startswith(('http://', 'https://')) else f'https://{company_url}'

          # Comprehensive phone number extraction patterns
          phone_patterns = [
                r'\+?1?\s*\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})',  # Formatted phone numbers
                r'\d{3}[-.]?\d{3}[-.]?\d{4}'  # Simple phone number pattern
            ]

          # Try extracting from URL
          for pattern in phone_patterns:
              matches = re.findall(pattern, company_url)
              if matches:
                  if isinstance(matches[0], tuple):
                      # Handle grouped matches
                      return f"{matches[0][0]}-{matches[0][1]}-{matches[0][2]}"
                  else:
                      # Handle simple matches
                      phone = re.sub(r'[^\d]', '', matches[0])
                      if 10 <= len(phone) <= 11:
                          return f"{phone[-10:-7]}-{phone[-7:-4]}-{phone[-4:]}"

          # If no phone found, return a generic contact indicator
          return f"Contact via: {urlparse(company_url).netloc}"


        except Exception as e:
            logging.error(f"Phone extraction error for {company_url}: {e}")
            return f"Contact via: {company_url}"

    def log_error(self, error_type, error_message):
        """
        Enhanced error logging method.

        :param error_type: Type of error
        :param error_message: Detailed error message
        """
        try:
            # Log to file
            logging.error(error_message)

            # Additional file logging
            with open("enhanced_email_validation_errors.txt", "a") as log_file:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"[{timestamp}] {error_type}: {error_message}\n")
        except Exception as e:
            print(f"Failed to log error: {e}")

    def process_row(self, row):
        """
        Comprehensive row processing with robust error handling.

        :param row: DataFrame row to process
        :return: Processed row data
        """
        try:
            name = HumanName(str(row.get("Full Name", "")))
            first_name = name.first or "Unknown"
            last_name = name.last or "Contact"
            company_url = str(row.get("Company URL", "")).strip()
            pattern = row.get("Pattern")

            # More forgiving input validation
            if not company_url:
                return {
                    "Email": None,
                    "Phone Number": "No contact information",
                    "Email Status": "No URL provided"
                }

            # Generate and verify email
            email = self.generate_emails(first_name, last_name, company_url, pattern)
            email_status = "Valid" if self.verify_email(email) else "Invalid"

            # Extract phone number with fallback
            phone = self.extract_phone_number(company_url)

            return {
                "Email": email,
                "Phone Number": phone,
                "Email Status": email_status
            }

        except Exception as e:
            logging.error(f"Error processing row: {e}")
            return {
                "Email": None,
                "Phone Number": "Contact information extraction failed",
                "Email Status": "Processing Error"
            }

    def handle_batch_processing(self):
        """
        Enhanced batch processing with comprehensive error handling and reporting.
        """
        try:
            self.console.print("[cyan]Please upload Excel files for processing...[/cyan]")
            uploaded_files = files.upload()

            if not uploaded_files:
                self.console.print("[red]No files uploaded. Exiting...[/red]")
                return

            all_results = []

            with Progress() as progress:
                for filename, file_content in uploaded_files.items():
                    task = progress.add_task(f"Processing {filename}...", total=100)

                    try:
                        # Save the uploaded file locally
                        local_filename = f"temp_{filename}"
                        with open(local_filename, 'wb') as f:
                            f.write(file_content)

                        # Read the Excel file
                        df = pd.read_excel(local_filename)

                        # Ensure required columns exist
                        required_columns = ["Full Name", "Company URL"]
                        for col in required_columns:
                            if col not in df.columns:
                                df[col] = ""

                        # Process rows
                        results = []
                        for _, row in df.iterrows():
                            result = self.process_row(row)
                            results.append(result)
                            progress.update(task, advance=100/len(df))

                        # Combine original data with results
                        results_df = pd.DataFrame(results)
                        output_df = pd.concat([df, results_df], axis=1)
                        all_results.append(output_df)

                        # Remove temporary file
                        os.remove(local_filename)
                        progress.update(task, completed=100)

                    except Exception as e:
                        self.console.print(f"[red]Error processing {filename}: {e}[/red]")
                        logging.error(f"Error processing file {filename}: {e}")

            # Combine all results
            if all_results:
                final_results = pd.concat(all_results, ignore_index=True)

                # Save results
                output_filename = "Verified_Results.xlsx"
                final_results.to_excel(output_filename, index=False)

                # Ensure file download
                try:
                    files.download(output_filename)
                    self.console.print(f"[green]Results saved to {output_filename}[/green]")
                except Exception as download_error:
                    self.console.print(f"[yellow]Warning: Could not use files.download(). Please check the file {output_filename}[/yellow]")
                    logging.error(f"File download error: {download_error}")

            else:
                self.console.print("[red]No results to process[/red]")

        except Exception as e:
            logging.error(f"Unhandled error in batch processing: {e}")
            self.console.print(f"[red]An error occurred during batch processing: {e}[/red]")

    def __del__(self):
        """Cleanup on object destruction."""
        self.save_cached_urls()
        self.session.close()

def main():
    """Main execution point for the Enhanced Email Verifier."""
    verifier = EnhancedEmailVerifier()
    verifier.handle_batch_processing()

if __name__ == "__main__":
    main()
