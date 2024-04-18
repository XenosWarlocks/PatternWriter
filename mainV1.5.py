# Public Version 1.5 (V 1.5)
import os
import pandas as pd
from urllib.parse import urlparse
from google.colab import files
from tqdm.notebook import tqdm
from datetime import date
import re

try:
    from email_validator import validate_email, EmailNotValidError
except ModuleNotFoundError:
    print("Please go to the first cell and install the 'email_validator' dependency.")

# V 1.5
def ask_for_email_validation():
    while True:
        response = input("Would you like to validate email addresses? (yes/no): ").lower()
        if response in ['yes', 'no']:
            return response == 'yes'
        else:
            print("Invalid response. Please enter 'yes' or 'no'.")

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
