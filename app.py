# PatternWriter with Enhanced Security and Error Handling
# Production Version 4.0 - Enhanced Security, Rate Limiting, and Error Handling

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
from requests.exceptions import Timeout, RequestException, ConnectionError, HTTPError
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Set, Any, Union
from datetime import datetime, timedelta
from collections import defaultdict
import ssl
import socket
from urllib.robotparser import RobotFileParser


class RateLimiter:
    """Rate limiter to prevent API abuse and respect service limits."""

    def __init__(self, max_calls: int = 100, time_window: int = 3600):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum calls allowed in time window
            time_window: Time window in seconds (default: 1 hour)
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self.last_call_time = 0

    def can_make_call(self) -> bool:
        """Check if a call can be made without exceeding rate limit."""
        now = time.time()

        # Remove calls outside the time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]

        # Check if we're under the limit
        return len(self.calls) < self.max_calls

    def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limits."""
        if not self.can_make_call():
            oldest_call = min(self.calls) if self.calls else time.time()
            wait_time = self.time_window - (time.time() - oldest_call) + 1
            if wait_time > 0:
                logging.info(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)

        # Add minimum delay between calls
        now = time.time()
        if self.last_call_time > 0:
            time_since_last = now - self.last_call_time
            min_interval = 1.0  # Minimum 1 second between calls
            if time_since_last < min_interval:
                time.sleep(min_interval - time_since_last)

        self.calls.append(time.time())
        self.last_call_time = time.time()


class SecurityValidator:
    """Security validator for URLs and inputs."""

    DANGEROUS_DOMAINS = {
        'localhost', '127.0.0.1', '0.0.0.0', '::1',
        '10.', '172.16.', '172.17.', '172.18.', '172.19.',
        '172.20.', '172.21.', '172.22.', '172.23.', '172.24.',
        '172.25.', '172.26.', '172.27.', '172.28.', '172.29.',
        '172.30.', '172.31.', '192.168.'
    }

    @classmethod
    def is_safe_domain(cls, domain: str) -> bool:
        """Check if domain is safe to access."""
        if not domain:
            return False

        domain = domain.lower().strip()

        # Check for dangerous patterns
        for dangerous in cls.DANGEROUS_DOMAINS:
            if domain.startswith(dangerous):
                return False

        # Basic domain validation
        if not re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', domain):
            return False

        return True

    @classmethod
    def sanitize_input(cls, input_str: str, max_length: int = 255) -> str:
        """Sanitize user input."""
        if not input_str:
            return ""

        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\'\x00-\x1f\x7f-\x9f]', '', str(input_str))

        # Limit length
        return sanitized[:max_length].strip()


class EmailVerifier:
    """A secure, production-ready email verification and generation system."""

    def __init__(self, log_file: str = "email_validation.log", api_key: str = None):
        """
        Initialize the EmailVerifier class.

        Args:
            log_file: Path to the log file
            api_key: Verimail.io API key
        """
        # Initialize core components first
        self.visited_urls: Set[str] = set()
        self.console = Console()
        self.security_validator = SecurityValidator()  # Initialize this FIRST

        # Initialize rate limiters
        self.api_rate_limiter = RateLimiter(max_calls=100, time_window=3600)  # 100 calls per hour
        self.web_rate_limiter = RateLimiter(max_calls=50, time_window=300)    # 50 calls per 5 minutes

        # Set up logging with proper configuration (now security_validator exists)
        self.logger = self._setup_logging(log_file)

        # Request session with proper configuration
        self.session = self._create_session()

        # Configuration
        self.validate_emails = self._ask_for_email_validation()
        self.api_key = self._get_api_key() if self.validate_emails else None
        self.exception_names = self._get_exception_names() if self.validate_emails else []

        # Email patterns
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

        # Statistics tracking
        self.stats = {
            'emails_generated': 0,
            'emails_verified': 0,
            'api_calls_made': 0,
            'errors_encountered': 0,
            'phone_numbers_found': 0
        }

    def _setup_logging(self, log_file: str) -> logging.Logger:
        """Set up secure logging configuration."""
        # Validate log file path (now security_validator exists)
        log_file = self.security_validator.sanitize_input(log_file, 100)
        if not log_file.endswith('.log'):
            log_file += '.log'

        logger = logging.getLogger("EmailVerifier")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        try:
            # Create file handler with proper permissions
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.INFO)

            # Create secure formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            # Also add console handler for important messages
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        except Exception as e:
            print(f"Error setting up logging: {e}")
            # Fallback to console only
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            logger.addHandler(console_handler)

        return logger

    def _create_session(self) -> requests.Session:
        """Create a properly configured requests session."""
        session = requests.Session()

        # Set reasonable timeouts
        session.timeout = (10, 30)  # (connect, read) timeout

        # Set up retry strategy
        from urllib3.util.retry import Retry
        from requests.adapters import HTTPAdapter

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set security headers
        session.headers.update({
            'User-Agent': 'EmailVerifier/4.0 (Professional Email Verification Tool)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

        return session

    def _ask_for_email_validation(self) -> bool:
        """Ask user if they want to validate emails with enhanced input validation."""
        max_attempts = 3
        attempts = 0

        while attempts < max_attempts:
            try:
                response = input("Would you like to validate email addresses using Verimail API? (yes/no): ").lower().strip()
                response = self.security_validator.sanitize_input(response, 10)

                if response in ['yes', 'y', '1', 'true']:
                    return True
                elif response in ['no', 'n', '0', 'false']:
                    return False
                else:
                    attempts += 1
                    if attempts < max_attempts:
                        self.console.print(f"[yellow]Invalid response. Please enter 'yes' or 'no'. ({max_attempts - attempts} attempts remaining)")
                    else:
                        self.console.print("[yellow]Maximum attempts reached. Defaulting to 'no'.")
                        return False
            except (EOFError, KeyboardInterrupt):
                self.console.print("[yellow]Input cancelled. Defaulting to 'no'.")
                return False
            except Exception as e:
                self.logger.error(f"Error getting user input: {e}")
                return False

        return False

    def _get_api_key(self) -> Optional[str]:
        """Get and validate Verimail API key."""
        max_attempts = 3
        attempts = 0

        while attempts < max_attempts:
            try:
                api_key = input("Enter your Verimail.io API key: ").strip()
                api_key = self.security_validator.sanitize_input(api_key, 100)

                if not api_key:
                    attempts += 1
                    if attempts < max_attempts:
                        self.console.print(f"[yellow]API key cannot be empty. ({max_attempts - attempts} attempts remaining)")
                    continue

                # Basic API key validation
                if len(api_key) < 10 or not re.match(r'^[a-zA-Z0-9_-]+$', api_key):
                    attempts += 1
                    if attempts < max_attempts:
                        self.console.print(f"[yellow]Invalid API key format. ({max_attempts - attempts} attempts remaining)")
                    continue

                return api_key

            except (EOFError, KeyboardInterrupt):
                self.console.print("[yellow]Input cancelled.")
                return None
            except Exception as e:
                self.logger.error(f"Error getting API key: {e}")
                return None

        self.console.print("[red]Failed to get valid API key.")
        return None

    def _get_exception_names(self) -> List[str]:
        """Get list of names to skip verification for with validation."""
        try:
            response = input("Would you like to add exception names to skip verification? (yes/no): ").lower().strip()
            response = self.security_validator.sanitize_input(response, 10)

            if response not in ['yes', 'y', '1', 'true']:
                return []

            exceptions = []
            self.console.print("[cyan]Enter full names (one per line). Enter a blank line when finished:")

            while len(exceptions) < 100:  # Limit to prevent abuse
                try:
                    name = input().strip()
                    if not name:
                        break

                    # Validate and sanitize name
                    name = self.security_validator.sanitize_input(name, 100)
                    if name and len(name) >= 2:
                        exceptions.append(name)
                    else:
                        self.console.print("[yellow]Name too short or invalid. Skipping.")

                except (EOFError, KeyboardInterrupt):
                    break

            self.console.print(f"[green]Added {len(exceptions)} exception names")
            return exceptions

        except Exception as e:
            self.logger.error(f"Error getting exception names: {e}")
            return []

    def _extract_domain(self, company_url: str) -> Optional[str]:
        """Extract and validate domain from company URL."""
        try:
            # Sanitize input
            company_url = self.security_validator.sanitize_input(company_url, 500)
            if not company_url:
                return None

            # Add protocol if missing
            if not company_url.startswith(('https://', 'http://')):
                company_url = 'https://' + company_url

            # Parse URL
            parsed = urlparse(company_url)
            domain = parsed.netloc.lower()

            if not domain:
                return None

            # Remove common subdomains
            subdomains_to_remove = ['www.', 'mail.', 'webmail.', 'smtp.', 'pop.', 'imap.']
            for subdomain in subdomains_to_remove:
                if domain.startswith(subdomain):
                    domain = domain[len(subdomain):]

            # Security check
            if not self.security_validator.is_safe_domain(domain):
                self.logger.warning(f"Unsafe domain detected: {domain}")
                return None

            return domain

        except Exception as e:
            self.logger.error(f"Error extracting domain from {company_url}: {e}")
            return None

    def generate_emails(self, first_name: str, middle_name: str, last_name: str,
                       company_url: str, pattern: str) -> Optional[str]:
        """Generate email address with enhanced validation."""
        try:
            # Validate and sanitize inputs
            first_name = self.security_validator.sanitize_input(first_name, 50).lower()
            middle_name = self.security_validator.sanitize_input(middle_name or "", 50).lower()
            last_name = self.security_validator.sanitize_input(last_name, 50).lower()
            pattern = self.security_validator.sanitize_input(pattern, 50)

            if not first_name or not last_name:
                raise ValueError("First name and last name are required")

            # Validate names contain only letters and basic characters
            if not re.match(r'^[a-zA-Z\s\-\'\.]+$', first_name + last_name):
                raise ValueError("Names contain invalid characters")

            domain = self._extract_domain(company_url)
            if not domain:
                raise ValueError(f"Could not extract valid domain from URL: {company_url}")

            # Generate email prefix based on pattern
            email_prefix = self._generate_email_prefix(first_name, middle_name, last_name, pattern)
            if not email_prefix:
                raise ValueError(f"Could not generate email prefix for pattern: {pattern}")

            email = f"{email_prefix}@{domain}"

            # Validate final email format
            if not self._is_valid_email_format(email):
                raise ValueError(f"Generated email has invalid format: {email}")

            self.stats['emails_generated'] += 1
            return email

        except Exception as e:
            self.logger.error(f"Error generating email: {e}")
            self.stats['errors_encountered'] += 1
            return None

    def _generate_email_prefix(self, first_name: str, middle_name: str,
                              last_name: str, pattern: str) -> Optional[str]:
        """Generate email prefix based on pattern."""
        try:
            # Clean names - remove spaces and special characters
            first_clean = re.sub(r'[^a-z]', '', first_name)
            middle_clean = re.sub(r'[^a-z]', '', middle_name) if middle_name else ""
            last_clean = re.sub(r'[^a-z]', '', last_name)

            if not first_clean or not last_clean:
                return None

            patterns = {
                "{first}": first_clean,
                "{last}": last_clean,
                "{first}.{last}": f"{first_clean}.{last_clean}",
                "{first}{last}": f"{first_clean}{last_clean}",
                "{first}{middle}{last}": f"{first_clean}{middle_clean}{last_clean}",
                "{f}{last}": f"{first_clean[0]}{last_clean}",
                "{first}{l}": f"{first_clean}{last_clean[0]}",
                "{f}{l}": f"{first_clean[0]}{last_clean[0]}",
                "{l}{f}": f"{last_clean[0]}{first_clean[0]}",
                "{first}.{l}": f"{first_clean}.{last_clean[0]}",
                "{f}.{last}": f"{first_clean[0]}.{last_clean}",
                "{first}_{last}": f"{first_clean}_{last_clean}"
            }

            return patterns.get(pattern)

        except Exception as e:
            self.logger.error(f"Error generating email prefix: {e}")
            return None

    def _is_valid_email_format(self, email: str) -> bool:
        """Validate email format."""
        if not email or len(email) > 320:  # RFC 5321 limit
            return False

        # Basic email regex
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    def verify_email(self, email: str) -> Dict[str, Any]:
        """Verify email with enhanced error handling and security."""
        if not self.validate_emails:
            return {
                "valid": True,
                "status": "not_verified",
                "deliverable": None,
                "result": "validation_disabled",
                "full_response": None
            }

        if not self._is_valid_email_format(email):
            return {
                "valid": False,
                "status": "error",
                "deliverable": False,
                "result": "invalid_email_format",
                "full_response": None
            }

        # Check rate limit
        self.api_rate_limiter.wait_if_needed()

        try:
            self.logger.info(f"Verifying email: {email}")
            self.stats['api_calls_made'] += 1

            # Create secure HTTPS connection
            conn = http.client.HTTPSConnection(
                "api.verimail.io",
                timeout=30,
                context=ssl.create_default_context()
            )

            # Build secure query parameters
            params = {
                'email': email[:320],  # Limit email length
                'key': self.api_key
            }

            query_string = urlencode(params, safe='@')
            url = f'/v3/verify?{query_string}'

            # Make request with security headers
            headers = {
                'User-Agent': 'EmailVerifier/4.0',
                'Accept': 'application/json',
                'Connection': 'close'
            }

            conn.request("GET", url, headers=headers)
            response = conn.getresponse()
            data = response.read()
            conn.close()

            # Validate response
            if response.status != 200:
                self.logger.error(f"HTTP {response.status} error for email {email}")
                return {
                    "valid": False,
                    "status": "error",
                    "deliverable": False,
                    "result": f"http_error_{response.status}",
                    "full_response": None
                }

            # Parse JSON response
            try:
                json_data = json.loads(data.decode("utf-8"))
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error: {e}")
                return {
                    "valid": False,
                    "status": "error",
                    "deliverable": False,
                    "result": "json_decode_error",
                    "full_response": None
                }

            # Validate API response structure
            if not isinstance(json_data, dict):
                return {
                    "valid": False,
                    "status": "error",
                    "deliverable": False,
                    "result": "invalid_api_response",
                    "full_response": None
                }

            # Process results
            status = json_data.get('status', '').lower()
            result = json_data.get('result', '').lower()
            deliverable = json_data.get('deliverable', False)

            if status != 'success':
                self.logger.error(f"API error for {email}: {status}")
                return {
                    "valid": False,
                    "status": "error",
                    "deliverable": False,
                    "result": f"api_error_{status}",
                    "full_response": json_data
                }

            # Determine validity
            is_valid = (
                result == 'deliverable' or
                (deliverable and result in ['catch_all', 'inbox_full'])
            )

            if is_valid:
                self.stats['emails_verified'] += 1

            return {
                "valid": is_valid,
                "status": json_data.get('status'),
                "deliverable": json_data.get('deliverable'),
                "result": json_data.get('result'),
                "did_you_mean": json_data.get('did_you_mean', ''),
                "full_response": json_data
            }

        except Exception as e:
            self.logger.error(f"Error verifying email {email}: {e}")
            self.stats['errors_encountered'] += 1
            return {
                "valid": False,
                "status": "error",
                "deliverable": None,
                "result": f"verification_error: {str(e)[:100]}",
                "full_response": None
            }

    def extract_phone_number(self, company_url: str) -> Optional[str]:
        """Extract phone number with enhanced security and error handling."""
        # Sanitize and validate URL
        company_url = self.security_validator.sanitize_input(company_url, 500)
        if not company_url or company_url in self.visited_urls:
            return None

        # Check rate limit
        self.web_rate_limiter.wait_if_needed()

        # Add to visited URLs
        self.visited_urls.add(company_url)

        try:
            # Validate domain
            domain = self._extract_domain(company_url)
            if not domain or not self.security_validator.is_safe_domain(domain):
                self.logger.warning(f"Skipping unsafe domain: {domain}")
                return None

            # Add protocol if missing
            if not company_url.startswith(('https://', 'http://')):
                company_url = 'https://' + company_url

            # Check robots.txt compliance (basic check)
            try:
                rp = RobotFileParser()
                rp.set_url(f"{company_url}/robots.txt")
                rp.read()
                if not rp.can_fetch('*', company_url):
                    self.logger.info(f"Robots.txt disallows crawling: {company_url}")
                    return None
            except:
                pass  # Continue if robots.txt check fails

            # Make request with security measures
            headers = {
                'User-Agent': random.choice([
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                ]),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate'
            }

            response = self.session.get(
                company_url,
                headers=headers,
                timeout=(10, 30),
                allow_redirects=True,
                verify=True  # Verify SSL certificates
            )
            response.raise_for_status()

            # Limit response size to prevent memory issues
            if len(response.content) > 5 * 1024 * 1024:  # 5MB limit
                self.logger.warning(f"Response too large for {company_url}")
                return None

            # Parse with security considerations
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove potentially dangerous elements
            for tag in soup(['script', 'style', 'meta', 'link']):
                tag.decompose()

            # Search for phone numbers
            phone = self._find_phone_number(soup)
            if phone:
                self.stats['phone_numbers_found'] += 1
                return phone

            return None

        except (ConnectionError, Timeout, HTTPError) as e:
            self.logger.warning(f"Network error fetching {company_url}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error extracting phone from {company_url}: {e}")
            self.stats['errors_encountered'] += 1
            return None

    def _find_phone_number(self, soup: BeautifulSoup) -> Optional[str]:
        """Find phone number in parsed HTML."""
        try:
            # Enhanced phone number patterns
            phone_patterns = [
                r'\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',  # US format
                r'\+?([1-9]\d{0,3})?[-.\s]?\(?(\d{1,4})\)?[-.\s]?(\d{1,4})[-.\s]?(\d{1,4})',  # International
                r'(\d{3})-(\d{3})-(\d{4})',  # XXX-XXX-XXXX
                r'(\d{3})\.(\d{3})\.(\d{4})',  # XXX.XXX.XXXX
                r'\((\d{3})\)\s*(\d{3})-(\d{4})'  # (XXX) XXX-XXXX
            ]

            # Search in tel: links first (most reliable)
            tel_links = soup.find_all('a', href=re.compile(r'^tel:'))
            for link in tel_links:
                phone = link.get('href').replace('tel:', '').strip()
                if self._is_valid_phone(phone):
                    return self._format_phone(phone)

            # Search in specific elements
            phone_selectors = [
                '*[class*="phone"]',
                '*[class*="telephone"]',
                '*[class*="contact"]',
                '*[id*="phone"]',
                '*[id*="contact"]',
                'footer',
                '.contact-info',
                '.header-contact'
            ]

            for selector in phone_selectors:
                try:
                    elements = soup.select(selector)
                    for element in elements:
                        text = element.get_text(strip=True)
                        for pattern in phone_patterns:
                            matches = re.findall(pattern, text)
                            for match in matches:
                                if isinstance(match, tuple):
                                    phone = ''.join(match)
                                else:
                                    phone = match

                                if self._is_valid_phone(phone):
                                    return self._format_phone(phone)
                except:
                    continue

            return None

        except Exception as e:
            self.logger.error(f"Error finding phone number: {e}")
            return None

    def _is_valid_phone(self, phone: str) -> bool:
        """Enhanced phone number validation."""
        if not phone:
            return False

        # Extract only digits
        digits = re.sub(r'\D', '', phone)

        # Length validation (7-15 digits per international standards)
        if len(digits) < 7 or len(digits) > 15:
            return False

        # Reject obviously invalid patterns
        invalid_patterns = [
            '0000000', '1111111', '2222222', '3333333', '4444444',
            '5555555', '6666666', '7777777', '8888888', '9999999',
            '1234567', '7654321', '0123456', '9876543'
        ]

        return digits not in invalid_patterns

    def _format_phone(self, phone: str) -> str:
        """Format phone number consistently."""
        digits = re.sub(r'\D', '', phone)

        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits.startswith('1'):
            return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        else:
            return f"+{digits}"

    def _should_try_alternative_patterns(self, verification_result: Dict[str, Any]) -> bool:
        """Determine if alternative patterns should be tried."""
        if not verification_result:
            return False

        result = verification_result.get('result', '').lower()

        # Try alternatives for hard failures
        failure_indicators = ['hardbounce', 'undeliverable', 'invalid']
        return result in failure_indicators

    def _try_alternative_patterns(self, first_name: str, middle_name: str, last_name: str,
                                 company_url: str, current_pattern: str) -> Dict[str, Any]:
        """Try alternative email patterns with rate limiting."""
        alternative_patterns = [p for p in self.standard_patterns if p != current_pattern]
        attempted_patterns = [current_pattern]
        attempts_count = 1

        self.logger.info(f"Trying alternative patterns for {first_name} {last_name}")

        for pattern in alternative_patterns:
            if attempts_count >= 5:  # Limit attempts to prevent abuse
                break

            try:
                email = self.generate_emails(first_name, middle_name, last_name, company_url, pattern)
                if not email:
                    continue

                verification_result = self.verify_email(email)
                attempted_patterns.append(pattern)
                attempts_count += 1

                self.logger.info(f"Pattern {pattern} -> {email} -> {verification_result.get('result', 'unknown')}")

                # Return if we found a good result
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

                # Rate limiting between attempts
                time.sleep(random.uniform(1, 3))

            except Exception as e:
                self.logger.error(f"Error trying pattern {pattern}: {e}")
                continue

        return {
            "email": None,
            "verification": None,
            "pattern_used": None,
            "patterns_attempted": attempted_patterns,
            "attempts_count": attempts_count,
            "success": False
        }

    def _process_row(self, row: pd.Series, index: int) -> Tuple[int, Dict[str, Any]]:
        """Process a single row with enhanced error handling."""
        result = row.to_dict()

        try:
            # Validate required fields
            if 'Full Name' not in row or pd.isna(row['Full Name']) or not str(row['Full Name']).strip():
                raise ValueError("Missing or empty Full Name")

            if 'Company URL' not in row or pd.isna(row['Company URL']) or not str(row['Company URL']).strip():
                raise ValueError("Missing or empty Company URL")

            if 'Pattern' not in row or pd.isna(row['Pattern']) or not str(row['Pattern']).strip():
                raise ValueError("Missing or empty Pattern")

            # Parse and validate name
            full_name = self.security_validator.sanitize_input(str(row['Full Name']), 200)
            name = HumanName(full_name)

            result['First Name'] = name.first or ""
            result['Middle Name'] = name.middle or ""
            result['Last Name'] = name.last or ""

            if not result['First Name'] or not result['Last Name']:
                raise ValueError("Could not parse first and last name")

            # Validate company URL
            company_url = self.security_validator.sanitize_input(str(row['Company URL']), 500)
            if not self._extract_domain(company_url):
                raise ValueError("Invalid company URL or unsafe domain")

            # Generate email
            pattern = self.security_validator.sanitize_input(str(row['Pattern']), 50)
            initial_email = self.generate_emails(
                result['First Name'],
                result['Middle Name'],
                result['Last Name'],
                company_url,
                pattern
            )

            if not initial_email:
                raise ValueError("Could not generate email address")

            result['Email'] = initial_email

            # Handle verification
            if self.validate_emails and full_name in self.exception_names:
                # Skip verification for exception names
                result['Email Verification'] = None
                result['Verification Status'] = "skipped"
                result['Deliverable'] = None
                result['Verification Result'] = "Exception name"
                result['Patterns Attempted'] = pattern
                result['Patterns Count'] = 1
                result['Did You Mean'] = ""

            elif self.validate_emails:
                # Verify email
                verification_result = self.verify_email(initial_email)

                # Try alternatives if needed
                if self._should_try_alternative_patterns(verification_result):
                    self.logger.info(f"Trying alternatives for {initial_email}")

                    retry_result = self._try_alternative_patterns(
                        result['First Name'],
                        result['Middle Name'],
                        result['Last Name'],
                        company_url,
                        pattern
                    )

                    if retry_result['success']:
                        result['Email'] = retry_result['email']
                        verification_result = retry_result['verification']
                        result['Pattern'] = retry_result['pattern_used']

                    result['Patterns Attempted'] = ', '.join(retry_result['patterns_attempted'])
                    result['Patterns Count'] = retry_result['attempts_count']
                else:
                    result['Patterns Attempted'] = pattern
                    result['Patterns Count'] = 1

                # Store verification results
                result['Email Verification'] = verification_result.get('valid', False)
                result['Verification Status'] = verification_result.get('status', 'unknown')
                result['Deliverable'] = verification_result.get('deliverable')
                result['Verification Result'] = verification_result.get('result', 'unknown')
                result['Did You Mean'] = verification_result.get('did_you_mean', '')

            else:
                # No verification
                result['Email Verification'] = None
                result['Verification Status'] = "not_verified"
                result['Deliverable'] = None
                result['Verification Result'] = "Validation disabled"
                result['Patterns Attempted'] = pattern
                result['Patterns Count'] = 1
                result['Did You Mean'] = ""

            # Extract phone number (with rate limiting built in)
            try:
                result['Phone Number'] = self.extract_phone_number(company_url)
            except Exception as e:
                self.logger.warning(f"Could not extract phone for {company_url}: {e}")
                result['Phone Number'] = None

        except Exception as e:
            self.logger.error(f"Error processing row {index}: {e}")
            self.stats['errors_encountered'] += 1

            # Set error values
            result['Email'] = None
            result['Email Verification'] = False
            result['Verification Status'] = "error"
            result['Deliverable'] = False
            result['Verification Result'] = f"Processing error: {str(e)[:100]}"
            result['Patterns Attempted'] = None
            result['Patterns Count'] = 0
            result['Phone Number'] = None
            result['Did You Mean'] = ""

        return (index, result)

    def process_file(self, filename: str, progress_bar: tqdm) -> Optional[pd.DataFrame]:
        """Process file with enhanced validation and security."""
        try:
            # Validate filename
            filename = self.security_validator.sanitize_input(filename, 255)
            if not filename.endswith(('.xlsx', '.xls')):
                self.logger.error(f"Invalid file type: {filename}")
                return None

            # Check file size (limit to 50MB)
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                if file_size > 50 * 1024 * 1024:  # 50MB
                    self.logger.error(f"File too large: {filename} ({file_size} bytes)")
                    return None

            # Read with error handling
            try:
                df = pd.read_excel(filename, nrows=10000)  # Limit rows to prevent memory issues
            except Exception as e:
                self.logger.error(f"Error reading Excel file {filename}: {e}")
                return None

            # Validate columns
            required_columns = ['Full Name', 'Company URL', 'Pattern']
            missing_columns = set(required_columns) - set(df.columns)

            if missing_columns:
                self.logger.error(f"Missing required columns in {filename}: {missing_columns}")
                return None

            # Data validation and cleaning
            original_count = len(df)

            # Remove rows with missing required data
            df = df.dropna(subset=required_columns)

            # Remove duplicates
            df = df.drop_duplicates(subset=['Full Name', 'Company URL'])

            # Additional validation
            df = df[df['Full Name'].str.len() >= 3]  # Minimum name length
            df = df[df['Company URL'].str.len() >= 5]  # Minimum URL length

            cleaned_count = len(df)
            self.logger.info(f"File {filename}: {original_count} -> {cleaned_count} rows after cleaning")

            if df.empty:
                self.logger.warning(f"No valid data remaining in {filename}")
                return None

            # Process rows with controlled concurrency
            results = []
            max_workers = min(3, len(df))  # Limit workers to prevent rate limiting

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit jobs
                futures = {
                    executor.submit(self._process_row, row, i): i
                    for i, (_, row) in enumerate(df.iterrows())
                }

                indexed_results = []

                for future in as_completed(futures):
                    try:
                        original_index, result = future.result(timeout=120)  # 2-minute timeout per row
                        indexed_results.append((original_index, result))
                    except Exception as e:
                        self.logger.error(f"Timeout or error processing row: {e}")
                        continue

                # Sort by original index
                indexed_results.sort(key=lambda x: x[0])
                results = [result for _, result in indexed_results]

            if not results:
                self.logger.error(f"No results generated from {filename}")
                return None

            processed_df = pd.DataFrame(results)

            # Clean up input file
            try:
                os.remove(filename)
                self.logger.info(f"Deleted input file: {filename}")
            except Exception as e:
                self.logger.warning(f"Could not delete {filename}: {e}")

            progress_bar.update(1)
            return processed_df

        except Exception as e:
            self.logger.error(f"Error processing file {filename}: {e}")
            self.stats['errors_encountered'] += 1
            return None

    def _print_statistics(self) -> None:
        """Print processing statistics."""
        self.console.print("\n[cyan]Processing Statistics:")
        self.console.print(f"• Emails generated: {self.stats['emails_generated']}")
        self.console.print(f"• Emails verified: {self.stats['emails_verified']}")
        self.console.print(f"• API calls made: {self.stats['api_calls_made']}")
        self.console.print(f"• Phone numbers found: {self.stats['phone_numbers_found']}")
        self.console.print(f"• Errors encountered: {self.stats['errors_encountered']}")

    def handle_batch_upload(self) -> None:
        """Handle batch upload with enhanced security and error handling."""
        try:
            self.console.print("[cyan]Please upload Excel files (max 10 files, 50MB each):")
            uploaded_files = files.upload()

            if not uploaded_files:
                self.console.print("[yellow]No files uploaded.")
                return

            # Validate number of files
            if len(uploaded_files) > 10:
                self.console.print("[red]Too many files. Maximum 10 files allowed.")
                return

            # Validate file types and sizes
            valid_files = []
            for filename, content in uploaded_files.items():
                filename_clean = self.security_validator.sanitize_input(filename, 255)

                if not filename_clean.endswith(('.xlsx', '.xls')):
                    self.console.print(f"[yellow]Skipping invalid file type: {filename}")
                    continue

                if len(content) > 50 * 1024 * 1024:  # 50MB
                    self.console.print(f"[yellow]Skipping large file: {filename}")
                    continue

                valid_files.append(filename_clean)

            if not valid_files:
                self.console.print("[red]No valid files to process.")
                return

            # Process files
            aggregated_df = pd.DataFrame()
            progress_bar = tqdm(total=len(valid_files), desc="Processing files")

            for filename in valid_files:
                try:
                    processed_df = self.process_file(filename, progress_bar)
                    if processed_df is not None and not processed_df.empty:
                        aggregated_df = pd.concat([aggregated_df, processed_df], ignore_index=True)
                except Exception as e:
                    self.logger.error(f"Error processing {filename}: {e}")
                    continue

            progress_bar.close()

            if aggregated_df.empty:
                self.console.print("[red]No data was successfully processed.")
                return

            # Generate output filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f'Verified_Emails_{timestamp}.xlsx'

            # Save with error handling
            try:
                with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
                    aggregated_df.to_excel(writer, index=False, sheet_name='Results')

                    # Add statistics sheet
                    stats_df = pd.DataFrame([self.stats])
                    stats_df.to_excel(writer, index=False, sheet_name='Statistics')

                self.console.print(f"[green]Output saved: {output_filename}")

            except Exception as e:
                self.logger.error(f"Error saving output: {e}")
                self.console.print(f"[red]Error saving output: {e}")
                return

            # Print statistics
            self._print_statistics()

            # Download files
            try:
                files.download(output_filename)
                self.console.print("[green]Output file downloaded successfully")
            except Exception as e:
                self.console.print(f"[yellow]Could not download output file: {e}")

            # Download log file
            try:
                if self.logger.handlers:
                    log_file = self.logger.handlers[0].baseFilename
                    if os.path.exists(log_file):
                        files.download(log_file)
                        self.console.print("[green]Log file downloaded successfully")
            except Exception as e:
                self.console.print(f"[yellow]Could not download log file: {e}")

            self.console.print("[green]Batch processing completed successfully!")

        except Exception as e:
            self.logger.error(f"Critical error in batch processing: {e}")
            self.console.print(f"[red]Critical error occurred: {e}")

        finally:
            # Cleanup
            try:
                if hasattr(self, 'session'):
                    self.session.close()
            except:
                pass

    def __del__(self):
        """Cleanup resources on destruction."""
        try:
            if hasattr(self, 'session'):
                self.session.close()
        except:
            pass


# Main execution
if __name__ == "__main__":
    try:
        email_verifier = EmailVerifier()
        email_verifier.handle_batch_upload()
    except KeyboardInterrupt:
        print("\n[yellow]Process interrupted by user.")
    except Exception as e:
        print(f"[red]Fatal error: {e}")
    finally:
        print("[cyan]Program terminated.")
