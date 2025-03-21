# Developer
import os
import re
import time
import random
import logging
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm.notebook import tqdm
from google.colab import files
from nameparser import HumanName
from urllib.parse import urlparse
from rich.console import Console
from rich.progress import Progress, TaskID
from requests.exceptions import Timeout, RequestException
from email_validator import validate_email, EmailNotValidError
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Set, Any, Union


class EmailVerifier:
    """A class to verify and generate email addresses based on patterns."""

    def __init__(self, log_file: str = "email_validation.log"):
        """
        Initialize the EmailVerifier class.

        Args:
            log_file: Path to the log file
        """
        # Set up logging
        self.logger = self._setup_logging(log_file)
        self.visited_urls: Set[str] = set()  # Set to store visited URLs
        self.console = Console()
        # Ask if user wants to validate emails
        self.validate_emails = self._ask_for_email_validation()
        # Standard email patterns
        self.standard_patterns = [
            "{first}.{last}",
            "{first}{last}",
            "{f}{last}",
            "{first}{l}",
            "{f}.{last}",
            "{first}_{last}"
        ]

    def _setup_logging(self, log_file: str) -> logging.Logger:
        """
        Set up logging configuration.

        Args:
            log_file: Path to the log file

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger("EmailVerifier")
        logger.setLevel(logging.INFO)

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(file_handler)

        return logger

    def _ask_for_email_validation(self) -> bool:
        """
        Ask user if they want to validate emails.

        Returns:
            Boolean indicating whether to validate emails
        """
        with Progress() as progress:
            task = progress.add_task("[cyan]Please answer the following question:", total=1)
            while True:
                response = input("Would you like to validate email addresses? (yes/no): ").lower()
                if response in ['yes', 'no']:
                    progress.update(task, advance=1, description="[green]Question answered", total=1)
                    return response == 'yes'
                else:
                    self.console.print("[red]Invalid response. Please enter 'yes' or 'no'.")

    def _extract_domain(self, company_url: str) -> str:
        """
        Extract domain from a company URL.

        Args:
            company_url: URL of the company

        Returns:
            Extracted domain name
        """
        # Adding protocol if missing
        if not company_url.startswith('https://') and not company_url.startswith('http://'):
            company_url = 'https://' + company_url

        # Extract domain names from company URL
        domain = urlparse(company_url).netloc
        return domain.replace('www.', '')

    def generate_emails(self, first_name: str, last_name: str, company_url: str,
                        pattern: str) -> str:
        """
        Generate email addresses based on pattern.

        Args:
            first_name: First name of the person
            last_name: Last name of the person
            company_url: URL of the company
            pattern: Email pattern to use

        Returns:
            Generated email address
        """
        domain = self._extract_domain(company_url)

        # Define a dictionary to map pattern placeholders to corresponding values
        placeholders = {
            '{first}': first_name.lower(),
            '{last}': last_name.lower(),
            '{f}': first_name[0].lower() if first_name else '',
            '{l}': last_name[0].lower() if last_name else '',
            '{first}.{last}': f"{first_name.lower()}.{last_name.lower()}",
            '{first}_{last}': f"{first_name.lower()}_{last_name.lower()}"
        }

        # Replace placeholders in the pattern with actual values
        email = pattern
        for placeholder, value in placeholders.items():
            email = email.replace(placeholder, value)

        return f"{email}@{domain}"

    def verify_email(self, email: str) -> bool:
        """
        Verify if an email address is valid.

        Args:
            email: Email address to verify

        Returns:
            Boolean indicating whether the email is valid
        """
        if not self.validate_emails:
            return True

        try:
            validate_email(email, check_deliverability=True, timeout=10)
            return True
        except EmailNotValidError as e:
            self.logger.warning(f"Email validation error for {email}: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error validating {email}: {str(e)}")
            return self._manual_validation(email)

    def _manual_validation(self, email: str) -> bool:
        """
        Perform basic manual validation of an email.

        Args:
            email: Email address to validate

        Returns:
            Boolean indicating whether the email is valid
        """
        # Check if email contains "@" and at least one "." after "@"
        if "@" in email:
            domain = email.split("@")[1]
            return "." in domain
        return False

    def extract_phone_number(self, company_url: str) -> Optional[str]:
        """
        Extract phone number from a company website.

        Args:
            company_url: URL of the company website

        Returns:
            Extracted phone number or None
        """
        # Check if the URL has already been visited
        if company_url in self.visited_urls:
            return None

        # Add the URL to the set of visited URLs
        self.visited_urls.add(company_url)

        try:
            # Check and add missing protocol
            if not company_url.startswith('https://') and not company_url.startswith('http://'):
                company_url = 'https://' + company_url

            # Set up a list of user agents
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:86.0) Gecko/20100101 Firefox/86.0',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
            ]

            # Select a random user agent
            headers = {'User-Agent': random.choice(user_agents)}

            # Introduce a random delay between requests
            time.sleep(random.uniform(1, 3))

            # Visit the website with a timeout
            try:
                response = requests.get(company_url, timeout=10, headers=headers)
                response.raise_for_status()  # Raise an exception for bad status codes
                self.logger.info(f"Successfully fetched website: {company_url}")
            except Timeout:
                self.logger.warning(f"Timeout occurred while fetching {company_url}")
                return None
            except RequestException as e:
                self.logger.error(f"Error fetching {company_url}: {str(e)}")
                return None

            # Parse the HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Search for phone numbers using a more comprehensive regular expression pattern
            for tag in soup.find_all(["a", "p", "span", "div"], string=True):
                # Regular expression pattern for international phone numbers
                # pattern = r'(?:\+\d{1,3}[-.\s]?)?(?:\(?\d{1,4}\)?[-.\s]?)?(?:\d{1,4}[-.\s]?){1,4}\d{1,4}'
                pattern = r'^(?:\+\d{1,3}|0\d{1,3}|00\d{1,2})?(?:\s?\(\d+\))?(?:[-\/\s.]|\d)+$'
                matches = re.findall(pattern, tag.get_text())

                for match in matches:
                    # Validate the phone number (must be at least 7 digits)
                    digits = re.sub(r'\D', '', match)
                    if len(digits) >= 7:
                        return match.strip()

            # No valid phone number found
            return None

        except Exception as e:
            self.logger.error(f"Error extracting phone number for {company_url}: {str(e)}")
            return None

    def _process_row(self, row: pd.Series) -> Dict[str, Any]:
        """
        Process a single row from the dataframe.

        Args:
            row: A pandas Series containing the row data

        Returns:
            Dictionary with processed data
        """
        result = row.to_dict()

        # Parse full name
        name = HumanName(row['Full Name'])
        result['First Name'] = name.first
        result['Last Name'] = name.last

        # Generate email
        result['Email'] = self.generate_emails(
            name.first,
            name.last,
            row['Company URL'],
            row['Pattern']
        )

        # Verify email
        if self.validate_emails:
            result['Email Verification'] = self.verify_email(result['Email'])

        # Extract phone number
        result['Phone Number'] = self.extract_phone_number(row['Company URL'])

        return result

    def process_file(self, filename: str, progress_bar: tqdm) -> Optional[pd.DataFrame]:
        """
        Process a single Excel file.

        Args:
            filename: Name of the Excel file
            progress_bar: tqdm progress bar

        Returns:
            Processed DataFrame or None if processing failed
        """
        try:
            # Read the excel file
            df = pd.read_excel(filename)

            # Validate the columns in the dataframe
            required_columns = ['Full Name', 'Company URL', 'Pattern']
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                self.logger.error(f"Error: {filename} is missing required columns: {missing_columns}")
                return None

            # Process each row
            results = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(self._process_row, row): i for i, (_, row) in enumerate(df.iterrows())}

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Error processing row: {str(e)}")

            # Create a new DataFrame from the results
            processed_df = pd.DataFrame(results)

            # Delete input file
            os.remove(filename)
            self.logger.info(f"Deleted input file: {filename}")

            # Update progress bar
            progress_bar.update(1)

            return processed_df

        except Exception as e:
            self.logger.error(f"Error processing file {filename}: {str(e)}")
            return None

    def handle_batch_upload(self) -> None:
        """
        Handle batch upload of Excel files.
        """
        try:
            # Prompt user to upload input Excel files
            self.console.print("[cyan]Please upload the input Excel files:")
            uploaded_files = files.upload()

            # Check if any files were uploaded
            if not uploaded_files:
                self.console.print("[yellow]No files uploaded. Processing canceled.")
                return

            # Create an empty dataframe to hold aggregated results
            aggregated_df = pd.DataFrame()

            # Initialize tqdm progress bar
            progress_bar = tqdm(total=len(uploaded_files), desc="Processing files")

            # Process each uploaded file
            for filename in uploaded_files:
                processed_df = self.process_file(filename, progress_bar)
                if processed_df is not None:
                    aggregated_df = pd.concat([aggregated_df, processed_df], ignore_index=True)

            # Close progress bar
            progress_bar.close()

            # Check if any data was processed
            if aggregated_df.empty:
                self.console.print("[red]No data was successfully processed.")
                return

            # Save the aggregated dataframe to a single output file
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f'Verified_Emails_{timestamp}.xlsx'
            aggregated_df.to_excel(output_filename, index=False)

            # Download the output file
            self.console.print(f"[green]Downloading output file: {output_filename}")
            files.download(output_filename)

            # Download the log file
            log_file = self.logger.handlers[0].baseFilename
            self.console.print(f"[green]Downloading log file: {log_file}")
            files.download(log_file)

            self.console.print("[green]Batch processing completed successfully.")

        except Exception as e:
            self.logger.error(f"Error in batch processing: {str(e)}")
            self.console.print(f"[red]An error occurred during batch processing: {str(e)}")


# Process batch upload
if __name__ == "__main__":
    email_verifier = EmailVerifier()
    email_verifier.handle_batch_upload()
