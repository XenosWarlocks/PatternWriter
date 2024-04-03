  import os
  import pandas as pd
  from urllib.parse import urlparse
  from email_validator import validate_email, EmailNotValidError
  from socket import gaierror
  from google.colab import files
  from tqdm.notebook import tqdm

  class EmailVerifier:
      def __init__(self):
          self.errors_log_file = "email_validation_errors.txt"

      def generate_emails(self, first_name, last_name, company_url, pattern):
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

      def verify_email(self, email):
          try:
            validate_email(email, check_deliverability=True, timeout=10)
            return True
          except Exception as e:
            print(f"Email validation error: {type(e)}")  # Print the exception type
            if isinstance(e, EmailNotValidError):
              return False
            # Handle other exception types here (gaierror, BlockingIOError)
            return self.manual_validation(email)

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
                  required_columns = ['First', 'Last', 'Company URL', 'Pattern']
                  if not set(required_columns).issubset(df.columns):
                      print(f"Error: {filename} is missing required columns.")
                      continue

                  # Apply the function to create a new column for emails
                  df['Email'] = df.apply(
                      lambda row: self.generate_emails(
                          row['First'],
                          row['Last'],
                          row['Company URL'],
                          row['Pattern']
                      ), axis=1
                  )

                  # Verify email addresses
                  df['Email Verification'] = df['Email'].apply(self.verify_email)

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

  # Process batch upload
  email_verifier = EmailVerifier()
  email_verifier.handle_batch_upload()
