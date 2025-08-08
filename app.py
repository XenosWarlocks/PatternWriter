# PatternWriter with Fixed Verimail API Integration
# Public Version 3.4 - Verimail API Bug Fix Applied

import os
import re
import time
import random
import logging
import json
import http.client
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm.notebook import tqdm
from google.colab import files
from nameparser import HumanName
from urllib.parse import urlparse, urlencode
from rich.console import Console
from rich.progress import Progress, TaskID
from requests.exceptions import Timeout, RequestException
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Set, Any, Union

class EmailVerifier:
    """A class to verify and generate email addresses based on patterns."""
    def __init__(self, log_file: str = "email_validation.log", api_key: str = None):
        """
        Initialize the EmailVerifier class.

        Args:
            log_file: Path to the log file
            api_key: Verimail.io API key
        """
        # Set up logging
        self.logger = self._setup_logging(log_file)
        self.visited_urls: Set[str] = set()  # Set to store visited URLs
        self.console = Console()
        # Ask if user wants to validate emails
        self.validate_emails = self._ask_for_email_validation()
        # Get API key if validation is requested
        self.api_key = self._get_api_key() if self.validate_emails else None
        # Standard email patterns
        self.standard_patterns = [
            "{first}",
            "{last}",
            "{first}.{last}",
            "{first}{last}",
            "{f}{last}",
            "{first}{l}",
            "{f}.{last}",
            "{first}_{last}",
            "{first}{middle}{last}"
        ]
        # Get exception names for verification skipping
        self.exception_names = self._get_exception_names() if self.validate_emails else []

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

        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()

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
                response = input("Would you like to validate email addresses using Verimail API? (yes/no): ").lower()
                if response in ['yes', 'no']:
                    progress.update(task, advance=1, description="[green]Question answered", total=1)
                    return response == 'yes'
                else:
                    self.console.print("[red]Invalid response. Please enter 'yes' or 'no'.")

    def _get_api_key(self) -> str:
        """
        Get Verimail API key from user.

        Returns:
            API key string
        """
        with Progress() as progress:
            task = progress.add_task("[cyan]Please provide your Verimail.io API key:", total=1)
            api_key = input("Enter your Verimail.io API key: ")
            progress.update(task, advance=1, description="[green]API key received", total=1)
            return api_key

    def _get_exception_names(self) -> List[str]:
        """
        Get list of names to skip verification for.

        Returns:
            List of full names to skip verification
        """
        with Progress() as progress:
            task = progress.add_task("[cyan]Would you like to add exception names to skip verification?", total=1)
            response = input("Would you like to add exception names to skip verification? (yes/no): ").lower()

            if response == 'yes':
                exceptions = []
                self.console.print("[cyan]Enter full names (one per line). Enter a blank line when finished:")
                while True:
                    name = input()
                    if not name:
                        break
                    exceptions.append(name.strip())
                progress.update(task, advance=1, description=f"[green]Added {len(exceptions)} exception names", total=1)
                return exceptions
            else:
                progress.update(task, advance=1, description="[green]No exceptions added", total=1)
                return []

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

        # Remove common subdomains
        remove_subdomains = ["www.", "developer.", "aws.", "ai."]
        for sub in remove_subdomains:
            domain = domain.replace(sub, '')

        return domain

    def generate_emails(self, first_name: str, middle_name: str, last_name: str, company_url: str, pattern: str) -> str:
        """
        Generate email addresses based on pattern.

        Args:
            first_name: First name of the person
            middle_name: Middle name of the person
            last_name: Last name of the person
            company_url: URL of the company
            pattern: Email pattern to use

        Returns:
            Generated email address
        """
        # Validate inputs
        if not first_name or not last_name:
            raise ValueError("First name and last name cannot be empty")

        domain = self._extract_domain(company_url)

        if not domain:
            raise ValueError(f"Could not extract domain from URL: {company_url}")

        # Clean and normalize names
        first_name = first_name.strip().lower()
        middle_name = middle_name.strip().lower() if middle_name else ""
        last_name = last_name.strip().lower()

        # Generate email based on pattern
        if pattern == "{first}":
            email_prefix = first_name
        elif pattern == "{last}":
            email_prefix = last_name
        elif pattern == "{first}.{last}":
            email_prefix = f"{first_name}.{last_name}"
        elif pattern == "{first}{last}":
            email_prefix = f"{first_name}{last_name}"
        elif pattern == "{first}{middle}{last}":
            email_prefix = f"{first_name}{middle_name}{last_name}"
        elif pattern == "{f}{last}":
            email_prefix = f"{first_name[0]}{last_name}"
        elif pattern == "{first}{l}":
            email_prefix = f"{first_name}{last_name[0]}"
        elif pattern == "{f}{l}":
            email_prefix = f"{first_name[0]}{last_name[0]}"
        elif pattern == "{l}{f}":
            email_prefix = f"{last_name[0]}{first_name[0]}"
        elif pattern == "{first}.{l}":
            email_prefix = f"{first_name}.{last_name[0]}"
        elif pattern == "{f}.{last}":
            email_prefix = f"{first_name[0]}.{last_name}"
        elif pattern == "{first}_{last}":
            email_prefix = f"{first_name}_{last_name}"
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        return f"{email_prefix}@{domain}"

    def verify_email(self, email: str) -> Dict[str, Any]:
        """
        Verify if an email address is valid using Verimail.io API.

        Args:
            email: Email address to verify

        Returns:
            Dictionary containing verification results
        """
        if not self.validate_emails:
            return {
                "valid": True,
                "status": "not_verified",
                "deliverable": None,
                "result": "validation_disabled",
                "full_response": None
            }

        if not email or '@' not in email:
            return {
                "valid": False,
                "status": "error",
                "deliverable": False,
                "result": "invalid_email_format",
                "full_response": None
            }

        try:
            self.logger.info(f"Verifying email: {email}")

            # Use Verimail.io API for verification - FIXED URL FORMAT
            conn = http.client.HTTPSConnection("api.verimail.io")

            # Build query parameters correctly
            params = {
                'email': email,
                'key': self.api_key
            }
            query_string = urlencode(params)
            url = f'/v3/verify?{query_string}'

            # Add a random delay to avoid rate limiting
            time.sleep(random.uniform(0.5, 2))

            conn.request("GET", url)
            res = conn.getresponse()
            datajs = res.read()
            conn.close()  # Properly close connection

            # Check HTTP status code first
            if res.status != 200:
                self.logger.error(f"HTTP {res.status} error for email {email}")
                return {
                    "valid": False,
                    "status": "error",
                    "deliverable": False,
                    "result": f"http_error_{res.status}",
                    "full_response": None
                }

            # Parse the response
            data = json.loads(datajs.decode("utf-8"))

            self.logger.info(f"Verification result for {email}: {data.get('result', 'unknown')}")

            # Check if API request was successful
            if data.get('status') != 'success':
                self.logger.error(f"API error for {email}: {data}")
                return {
                    "valid": False,
                    "status": "error",
                    "deliverable": False,
                    "result": f"api_error: {data.get('status', 'unknown')}",
                    "full_response": data
                }

            # Determine if email is valid based on API response
            result = data.get('result', '').lower()
            deliverable = data.get('deliverable', False)

            # deliverable, inbox_full, hardbounce, softbounce, catch_all, disposable, undeliverable
            is_valid = (
                result == 'deliverable' or
                (deliverable and result in ['catch_all', 'inbox_full'])
            )

            # Return complete verification data
            return {
                "valid": is_valid,
                "status": data.get('status'),
                "deliverable": data.get('deliverable'),
                "result": data.get('result'),
                "did_you_mean": data.get('did_you_mean', ''),
                "full_response": data
            }

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error for email {email}: {str(e)}")
            return {
                "valid": False,
                "status": "error",
                "deliverable": None,
                "result": f"json_error: {str(e)}",
                "full_response": None
            }
        except Exception as e:
            self.logger.error(f"Error verifying email {email}: {str(e)}")
            # Fallback to basic validation on error
            return {
                "valid": self._manual_validation(email),
                "status": "error",
                "deliverable": None,
                "result": f"api_error: {str(e)}",
                "full_response": None
            }

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
            parts = email.split("@")
            if len(parts) == 2:
                domain = parts[1]
                return "." in domain and len(domain) > 2
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

            # Search for phone numbers using a more comprehensive regex pattern
            phone_pattern = r'(\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}|\+?[1-9]\d{1,14})'

            # Search in common phone number containers
            phone_selectors = [
                'a[href^="tel:"]',
                '*[class*="phone"]',
                '*[class*="contact"]',
                '*[id*="phone"]',
                '*[id*="contact"]'
            ]

            for selector in phone_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(strip=True)
                    if 'href' in element.attrs and element.attrs['href'].startswith('tel:'):
                        # Extract from tel: link
                        phone = element.attrs['href'].replace('tel:', '').strip()
                        if self._is_valid_phone(phone):
                            return phone

                    # Extract from text content
                    matches = re.findall(phone_pattern, text)
                    for match in matches:
                        if self._is_valid_phone(match):
                            return match.strip()

            # Fallback: search entire page text
            page_text = soup.get_text()
            matches = re.findall(phone_pattern, page_text)
            for match in matches:
                if self._is_valid_phone(match):
                    return match.strip()

            # No valid phone number found
            return None

        except Exception as e:
            self.logger.error(f"Error extracting phone number for {company_url}: {str(e)}")
            return None

    def _is_valid_phone(self, phone: str) -> bool:
        """
        Validate if a phone number is reasonable.

        Args:
            phone: Phone number string to validate

        Returns:
            Boolean indicating if phone number is valid
        """
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', phone)

        # Must have at least 7 digits and at most 15 (international standard)
        if len(digits) < 7 or len(digits) > 15:
            return False

        # Avoid obviously invalid patterns
        invalid_patterns = ['0000000', '1111111', '1234567', '7777777']
        return digits not in invalid_patterns

    def _should_try_alternative_patterns(self, verification_result: Dict[str, Any]) -> bool:
        """
        Determine if alternative patterns should be tried based on verification result.

        Args:
            verification_result: Result from email verification

        Returns:
            Boolean indicating if alternative patterns should be tried
        """
        # Get the result and handle None values properly
        result = verification_result.get('result')
        if result is None:
            return False

        # Convert to lowercase for comparison
        result = result.lower()

        # Try alternatives for hard bounces and undeliverable emails
        # Based on Verimail API docs: deliverable, inbox_full, hardbounce, softbounce, catch_all, disposable, undeliverable
        failure_indicators = [
            'hardbounce',
            'undeliverable'
        ]

        return result in failure_indicators

    def _try_alternative_patterns(self, first_name: str, middle_name: str, last_name: str, company_url: str,
                                current_pattern: str) -> Dict[str, Any]:
        """
        Try alternative email patterns if current pattern results in failure.

        Args:
            first_name: First name of the person
            middle_name: Middle name of the person
            last_name: Last name of the person
            company_url: URL of the company
            current_pattern: Current pattern that failed

        Returns:
            Dictionary with verification results and pattern attempts info
        """
        # Skip current pattern as it was already tried
        alternative_patterns = [p for p in self.standard_patterns if p != current_pattern]

        # Track patterns attempted
        attempted_patterns = [current_pattern]
        attempts_count = 1

        self.logger.info(f"Trying alternative patterns for {first_name} {last_name} at {company_url}")

        # Try each alternative pattern
        for pattern in alternative_patterns:
            try:
                email = self.generate_emails(first_name, middle_name, last_name, company_url, pattern)
                verification_result = self.verify_email(email)
                attempted_patterns.append(pattern)
                attempts_count += 1

                # Log the attempt
                self.logger.info(f"Pattern {pattern} -> {email} -> {verification_result.get('result', 'unknown')}")

                # If we found a valid email or one that's not a hard failure, return it
                if verification_result.get('valid', False) or \
                   not self._should_try_alternative_patterns(verification_result):
                    return {
                        "email": email,
                        "verification": verification_result,
                        "pattern_used": pattern,
                        "patterns_attempted": attempted_patterns,
                        "attempts_count": attempts_count,
                        "success": True
                    }

                # Add delay between attempts to avoid rate limiting
                time.sleep(random.uniform(1, 2))

            except Exception as e:
                self.logger.error(f"Error trying pattern {pattern}: {str(e)}")
                continue

        # If we get here, all patterns failed
        self.logger.warning(f"All email patterns failed for {first_name} {last_name} at {company_url}")
        return {
            "email": None,
            "verification": None,
            "pattern_used": None,
            "patterns_attempted": attempted_patterns,
            "attempts_count": attempts_count,
            "success": False
        }

    def _process_row(self, row: pd.Series, index: int) -> Tuple[int, Dict[str, Any]]:
        """
        Process a single row from the dataframe.

        Args:
            row: A pandas Series containing the row data
            index: The original index of the row

        Returns:
            Tuple containing the original index and dictionary with processed data
        """
        result = row.to_dict()

        try:
            # Parse full name
            name = HumanName(row['Full Name'])
            result['First Name'] = name.first
            result['Middle Name'] = name.middle if name.middle else ""  # Handle None middle name
            result['Last Name'] = name.last

            # Validate that we have both first and last names
            if not result['First Name'] or not result['Last Name']:
                self.logger.warning(f"Missing name components for: {row['Full Name']}")
                result['Email'] = None
                result['Email Verification'] = False
                result['Verification Status'] = "error"
                result['Deliverable'] = False
                result['Verification Result'] = "Missing name components"
                result['Patterns Attempted'] = None
                result['Patterns Count'] = 0
                result['Phone Number'] = None
                return (index, result)

            # Generate initial email
            initial_email = self.generate_emails(
                result['First Name'],
                result['Middle Name'],  # Pass middle name (now guaranteed to be string)
                result['Last Name'],
                row['Company URL'],
                row['Pattern']
            )
            result['Email'] = initial_email

            # Check if this name is in exception list
            if self.validate_emails and row['Full Name'] in self.exception_names:
                self.logger.info(f"Skipping verification for exception name: {row['Full Name']}")
                result['Email Verification'] = None
                result['Verification Status'] = "skipped"
                result['Deliverable'] = None
                result['Verification Result'] = "Exception name"
                result['Patterns Attempted'] = row['Pattern']
                result['Patterns Count'] = 1
            # Verify email
            elif self.validate_emails:
                verification_result = self.verify_email(initial_email)

                # Check if we should try alternative patterns
                if self._should_try_alternative_patterns(verification_result):
                    self.logger.info(f"Primary pattern failed for {initial_email}, trying alternatives")

                    retry_result = self._try_alternative_patterns(
                        result['First Name'],
                        result['Middle Name'],  # Pass middle name
                        result['Last Name'],
                        row['Company URL'],
                        row['Pattern']
                    )

                    if retry_result['success']:
                        # Update with successful pattern
                        result['Email'] = retry_result['email']
                        verification_result = retry_result['verification']
                        result['Pattern'] = retry_result['pattern_used']

                    # Add retry information
                    result['Patterns Attempted'] = ', '.join(retry_result['patterns_attempted'])
                    result['Patterns Count'] = retry_result['attempts_count']
                else:
                    # No retry needed
                    result['Patterns Attempted'] = row['Pattern']
                    result['Patterns Count'] = 1

                # Store verification results
                result['Email Verification'] = verification_result.get('valid', False)
                result['Verification Status'] = verification_result.get('status', 'unknown')
                result['Deliverable'] = verification_result.get('deliverable')
                result['Verification Result'] = verification_result.get('result', 'unknown')
                result['Did You Mean'] = verification_result.get('did_you_mean', '')
            else:
                # If not validating emails
                result['Email Verification'] = None
                result['Verification Status'] = "not_verified"
                result['Deliverable'] = None
                result['Verification Result'] = "Validation disabled"
                result['Patterns Attempted'] = row['Pattern']
                result['Patterns Count'] = 1

            # Extract phone number
            result['Phone Number'] = self.extract_phone_number(row['Company URL'])

        except Exception as e:
            self.logger.error(f"Error processing row {index}: {str(e)}")
            result['Email'] = None
            result['Email Verification'] = False
            result['Verification Status'] = "error"
            result['Deliverable'] = False
            result['Verification Result'] = f"Processing error: {str(e)}"
            result['Patterns Attempted'] = None
            result['Patterns Count'] = 0
            result['Phone Number'] = None

        return (index, result)

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

            # Validate data quality
            df = df.dropna(subset=required_columns)
            if df.empty:
                self.logger.error(f"No valid data found in {filename}")
                return None

            # Process each row
            results = []
            with ThreadPoolExecutor(max_workers=3) as executor:  # Reduced to avoid rate limiting
                # Submit jobs with their original indices
                futures = {executor.submit(self._process_row, row, i): i
                          for i, (_, row) in enumerate(df.iterrows())}

                # Create a list to store results with their original indices
                indexed_results = []

                for future in as_completed(futures):
                    try:
                        original_index, result = future.result()
                        indexed_results.append((original_index, result))
                    except Exception as e:
                        self.logger.error(f"Error processing row: {str(e)}")

                # Sort results by original index to maintain original order
                indexed_results.sort(key=lambda x: x[0])
                # Extract just the results (without indices)
                results = [result for _, result in indexed_results]

            # Create a new DataFrame from the results
            processed_df = pd.DataFrame(results)

            # Delete input file
            try:
                os.remove(filename)
                self.logger.info(f"Deleted input file: {filename}")
            except Exception as e:
                self.logger.warning(f"Could not delete input file {filename}: {str(e)}")

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

            # Print summary statistics
            total_rows = len(aggregated_df)
            if self.validate_emails:
                verified_emails = len(aggregated_df[aggregated_df['Email Verification'] == True])
                self.console.print(f"[green]Summary: {verified_emails}/{total_rows} emails verified successfully")
            else:
                self.console.print(f"[green]Summary: {total_rows} emails generated (validation disabled)")

            # Download the output file
            self.console.print(f"[green]Downloading output file: {output_filename}")
            files.download(output_filename)

            # Download the log file
            if self.logger.handlers:
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
