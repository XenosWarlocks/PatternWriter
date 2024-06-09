class EmailVerifier:
    def __init__(self):
        self.errors_log_file = "email_validation_errors.txt"
        self.create_errors_log_file_if_not_exists()
        self.visited_urls = set()  # Set to store visited URLs
        self.validate_emails = self.ask_for_email_validation()

    def create_errors_log_file_if_not_exists(self):
        import os
        if not os.path.exists(self.errors_log_file):
            with open(self.errors_log_file, "w") as log_file:
                log_file.write("Error Log File\n")

    def ask_for_email_validation(self):
        from rich.progress import Progress
        with Progress() as progress:
            task = progress.add_task("[cyan]Please answer the following question:", total=1)
            while True:
                response = input("Would you like to validate email addresses? (yes/no): ").lower()
                if response in ['yes', 'no']:
                    progress.update(task, advance=1, description="[green]Question answered", total=1)
                    return response == 'yes'
                else:
                    print("Invalid response. Please enter 'yes' or 'no'.")

    def generate_emails(self, first_name, last_name, company_url, pattern):
        from urllib.parse import urlparse

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
            '{first}.{last}': f"{first_name.lower()}.{last_name.lower()}",
            '{first}_{last}': f"{first_name.lower()}_{last_name.lower()}"
        }

        # Replace placeholders in the pattern with actual values
        email = pattern
        for placeholder, value in placeholders.items():
            email = email.replace(placeholder, value)
        return f"{email}@{domain}"

    def verify_email(self, email):
        if self.validate_emails:
            try:
                from email_validator import validate_email, EmailNotValidError
                validate_email(email, check_deliverability=True, timeout=10)
                return True
            except Exception as e:
                print(f"Email validation error: {type(e)}")
                # Handle other exception types here (gaierror, BlockingIOError)
                return self.manual_validation(email)
        else:
            # Skip validation and return the email directly
            return email

    def manual_validation(self, email):
        # Check if email contains "@"
        if "@" in email:
            return True
        else:
            return False

    def log_error(self, error_type, error_message):
        with open(self.errors_log_file, "a") as log_file:
            log_file.write(f"{error_type}: {error_message}\n")

    def handle_batch_upload(self):
        try:
            # Prompt user to upload input Excel files
            print("Please upload the input Excel files:")
            from google.colab import files
            uploaded_files = files.upload()

            # Check if any files were uploaded
            if not uploaded_files:
                print("No files uploaded. Processing canceled.")
                return

            import pandas as pd
            from tqdm.notebook import tqdm

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

                from nameparser import HumanName
                # Parse full name into first and last names
                df[['First Name', 'Last Name']] = df['Full Name'].apply(
                    lambda x: pd.Series([HumanName(x).first, HumanName(x).last])
                )

                # Apply the function to create a new column for emails
                df['Email'] = df.apply(
                    lambda row: self.generate_emails(
                        row['First Name'],
                        row['Last Name'],
                        row['Company URL'],
                        row['Pattern']
                    ), axis=1
                )

                # Conditionally generate "Email Verification" column based on user preference
                if self.validate_emails:
                    df['Email Verification'] = df['Email'].apply(self.verify_email)
                else:
                    pass

                # Extract phone numbers from company websites
                df['Phone Number'] = df['Company URL'].apply(self.extract_phone_number)

                # Append the processed dataframe to the aggregated dataframe
                aggregated_df = pd.concat([aggregated_df, df])

                import os
                # Delete the uploaded file
                os.remove(filename)
                print(f"Deleted file: {filename}")

                # Update tqdm progress bar
                progress_bar.update(1)

            # Close tqdm progress bar
            progress_bar.close()

            # Save the aggregated dataframe to a single output file
            output_filename = 'Verified_Lemon_Batch.xlsx'
            aggregated_df.to_excel(output_filename, index=False)

            # Download the output file
            files.download(output_filename)

            # Download the error log file
            files.download(self.errors_log_file)

            print("Batch processing completed. Output saved successfully.")

            # Print message after file deletion is completed
            print("File deletion completed.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def extract_phone_number(self, company_url):
        import random
        import time
        import re
        import requests
        from bs4 import BeautifulSoup
        from requests.exceptions import Timeout

        try:
            # Check if the URL has already been visited
            if company_url in self.visited_urls:
                return None

            # Add the URL to the set of visited URLs
            self.visited_urls.add(company_url)

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
            user_agent = random.choice(user_agents)

            # Set up headers with the selected user agent
            headers = {'User-Agent': user_agent}

            # Introduce a random delay between requests
            time.sleep(random.uniform(1, 3))

            # Visit the website with a timeout (assuming company_url is valid)
            try:
                response = requests.get(company_url, timeout=10, headers=headers)
                response.raise_for_status()  # Raise an exception for bad status codes
                print(f"Successfully fetched website: {company_url}")  # Print success message
            except Timeout:
                print(f"Timeout occurred while fetching {company_url}. Moving on to the next website.")
                return None  # Return None on timeout
            except requests.exceptions.RequestException as e:
                print(f"Error fetching {company_url}: {e}")
                return None  # Return None on other errors

            # Parse the HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Search for phone numbers using a more comprehensive regular expression pattern
            phone_numbers = []
            for tag in soup.find_all(["a", "p", "span", "div"], string=True):
                # Regular expression pattern for international phone numbers
                # Previous Regex
                # pattern = r'\+?(?:[-\s()]?\d[-\s()]?){8,20}'
                pattern = r'(\+?\d{1,4}[-\s()]?)?(\(?\d{1,4}\)?[-\s.]?)?(\d{1,4}[-\s.]?){1,4}'
                matches = re.findall(pattern, tag.get_text())
                for match in matches:
                    # Append the phone number to the list
                    phone_numbers.append(match.strip())

            # Check if phone numbers were found
            if phone_numbers:
                # Return the first phone number found
                return phone_numbers[0]

            # No phone number found
            return None

        except Exception as e:
            error_message = f"Error extracting phone number for {company_url}: {str(e)}"
            print(error_message)
            self.log_error("ExtractionError", error_message)
            return None


# Process batch upload
email_verifier = EmailVerifier()
email_verifier.handle_batch_upload()
