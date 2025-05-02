# Public Version 2.0 (V 1.5)
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
from urllib.parse import urlparse
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
                response = input("Would you like to validate email addresses using Verimail API? (yes/no): ").lower()
                if response in ['yes', 'no']:
                    progress.update(task, advance=1, description="[green]Question answered", total=1)
                    return response == 'yes'
                else:
                    self.console.print("[red]Invalid response. Please enter 'yes' or 'no'.")
 # V 1.5

def generate_emails(first_name, last_name, company_url, pattern):
    # Adding protocol
    if not company_url.startswith('https://') and not company_url.startswith('http://'):
        company_url = 'https://' + company_url
    # Extract domain names from company URL
    domain = urlparse(company_url).netloc
    domain = domain.replace('www.', '')

    # Define a dictionary to map pattern placeholders to corresponding values
    placeholders = {
        '{first}': first_name.lower(),
        '{last}': last_name.lower(),
        '{f}': first_name[0].lower(),
        '{l}': last_name[0].lower(),
        '{first}.{last}': f"{first_name.lower()}.{last_name.lower()}"
    }

    # Replace placeholders in the pattern with actual values
    email = pattern
    for placeholder, value in placeholders.items():
        email = email.replace(placeholder, value)
    return f"{email}@{domain}"

def verify_email(email, validate_emails=True):
    if validate_emails:
        try:
            validate_email(email, check_deliverability=True, timeout=10)
            return True
        except Exception as e:
            print(f"Email validation error: {type(e)}")
            if isinstance(e, EmailNotValidError):
                return False
            # Handle other exception types here (gaierror, BlockingIOError)
            return manual_validation(email)
    else:
        # Skip validation and return True directly
        return True

def manual_validation(email):
    # Check if email contains "@"
    if "@" in email:
        return True
    else:
        return False

def handle_batch_upload(validate_emails=True):
    # Ask the user for email validation preference
    validate_emails = ask_for_email_validation()

    try:
        # Prompt user to upload the Excel files
        print("Please upload the Excel files:")
        uploaded_files = files.upload()

        # Check if any files were uploaded
        if not uploaded_files:
            print("No files uploaded. Batch processing canceled.")
            return

        # Create an empty dataframe to hold aggregated results
        aggregated_df = pd.DataFrame()

        # Initialize tqdm progress bar
        progress_bar = tqdm(total=len(uploaded_files), desc="Processing")

        # Iterate over uploaded files
        for filename, file_content in uploaded_files.items():
            # Read the uploaded excel file
            df = pd.read_excel(filename)

            # Validate the columns in the dataframe
            required_columns = ['Full Name', 'Company URL', 'Pattern']
            if not set(required_columns).issubset(df.columns):
                print(f"Error: {filename} is missing required columns.")
                continue

            # v.1.1
            # Extract first and last names
            df['First Name'] = df['Full Name'].str.extract(r'^(\S+)')
            df['Last Name'] = df['Full Name'].str.extract(r'(\S+)$')

            # Apply the function to create a new column for emails
            df['Email'] = df.apply(
                lambda row: generate_emails(
                    row['First Name'],
                    row['Last Name'],
                    row['Company URL'],
                    row['Pattern']
                ), axis=1
            )

            # V 1.5
            if validate_emails:
                df['Email Verification'] = df['Email'].apply(
                    lambda x: verify_email(x, validate_emails)
                    )
            else:
                # Skip email verification, no need for the column
                pass  # Placeholder to maintain code structure

            # Append the processed dataframe to the aggregated dataframe
            aggregated_df = pd.concat([aggregated_df, df])

            # Delete the uploaded file
            os.remove(filename)
            print(f"Deleted file: {filename}")

            # Update tqdm progress bar
            progress_bar.update(1)

        # Close tqdm progress bar
        progress_bar.close()

        # Save the aggregated dataframe to a single output file
        output_filename = 'Lemon_v1.5.xlsx'

        aggregated_df.to_excel(output_filename, index=False)

        # Download the output file
        files.download(output_filename)

        print("Batch processing completed. Output saved successfully.")

        # Print message after file deletion is completed
        print("File deletion completed.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Process batch upload
handle_batch_upload()
