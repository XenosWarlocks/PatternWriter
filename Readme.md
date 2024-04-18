# Email Verification Tool

This tool automates the process of verifying email addresses extracted from contact information provided in Excel files. It uses the email_validator library to validate email addresses and generates a report of verified email addresses along with their verification status.

## Table of Contents

- [Usage](#usage)
  - [Step 1: One-time Setup](#step-1-one-time-setup)
  - [Understanding the Code](#understanding-the-code)
  - [Issues with "__enter__" Error](#issues-with-enter-error)
- [Support](#support)

## Usage

### Step 1: One-time Setup

1. Run the `first_step.py` script to set up the environment and install necessary dependencies:

```
bash python first_step.py
```
This script installs `email_validator` Python package using pip.
Now you are ready to run the `mainV1.5.py`

### Understanding the Code:

The main script (`mainV1.5.py`) performs the following tasks:

  1. It reads input Excel files and extracts contact information columns such as first name, last name, company URL, and email pattern.

  2. Using this information, it generates potential email addresses by replacing placeholders in the email pattern.

  3. It then verifies each generated email address using the `verify_email` method.

  4. If the email verification fails due to the "enter" error, it attempts manual validation by checking if the email address contains "@".
  
  5. Errors encountered during the email validation process are logged in the `email_validation_errors.txt` file for further analysis.


### Issues with "__enter__" Error:

Error Handling:

The code attempts to handle potential errors during email validation. However, you might encounter issues like network connectivity problems, library bugs, or exceeding API rate limits.

Contributing and Fixing Errors:

This code is designed to be improvable. If you encounter errors or have suggestions for improvement, feel free to:

Check the error messages for specific details about the problem.
Refer to the comments within the code (# symbol) for explanations of different functions.
Consider the troubleshooting tips mentioned in the comments to address common errors (e.g., network issues, outdated libraries).
You can modify the code (especially the `verify_email` function in `mainV1.5.py`) to improve error handling or explore alternative email validation libraries like `yoyo` or `python-whois`.

# Support:
For any questions or assistance, please open an issue in the GitHub repository.

