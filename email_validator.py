# Colab-friendly Verimail v3 verifier — paste one email per line

import sys
import time
import io
from getpass import getpass
from typing import List
import threading

# Install ipywidgets in case Colab doesn't have it enabled
try:
    import requests, pandas as pd, IPython
    import ipywidgets as widgets
    from IPython.display import display, clear_output
except Exception:
    !pip install requests pandas ipywidgets -q
    import requests, pandas as pd, IPython
    import ipywidgets as widgets
    from IPython.display import display, clear_output

from google.colab import files

API_URL = "https://api.verimail.io/v3/verify"

def verify_email(api_key: str, email: str, timeout: int = 20):
    """
    Call Verimail /v3/verify for a single email.
    Returns dict with useful fields and raw response if available.
    """
    params = {"key": api_key, "email": email}
    try:
        resp = requests.get(API_URL, params=params, timeout=timeout)
    except requests.RequestException as e:
        return {"email": email, "status": "error", "error": f"request error: {e}"}
    # Try parse JSON
    try:
        j = resp.json()
    except Exception:
        return {"email": email, "status": "error", "http_status": resp.status_code, "error": resp.text}
    # Normalize output (Verimail returns fields like result, deliverable, etc.)
    out = {
        "email": j.get("email", email),
        "status": j.get("status") or ("success" if resp.status_code == 200 else "error"),
        "result": j.get("result"),
        "deliverable": j.get("deliverable"),
        "user": j.get("user"),
        "domain": j.get("domain"),
        "did_you_mean": j.get("did_you_mean", ""),
        "raw": j
    }
    return out

# UI widgets
api_key_widget = widgets.Password(
    description='API Key:',
    placeholder='Enter your Verimail API key (hidden)'
)
emails_text = widgets.Textarea(
    value='',
    placeholder='example1@gmail.com\nexample2@example.com\n...',
    description='Provide the emails for verification:',
    layout=widgets.Layout(width='100%', height='180px'),
    style={'description_width': 'initial'}
)
start_button = widgets.Button(description='Start Verification', button_style='primary')
cancel_button = widgets.Button(description='Cancel', button_style='danger')
progress_bar = widgets.IntProgress(value=0, min=0, max=1, description='Progress:')
status_out = widgets.Output(layout={'border': '1px solid #ddd'})
results_out = widgets.Output()
download_link_out = widgets.Output()

# Shared state
_stop_flag = threading.Event()

def run_verification(_):
    # Prevent double-clicks
    start_button.disabled = True
    cancel_button.disabled = False
    with status_out:
        clear_output(wait=True)
        print("Preparing...")

    api_key = api_key_widget.value.strip()
    if not api_key:
        with status_out:
            clear_output(wait=True)
            print("Please enter your Verimail API key first (top field).")
        start_button.disabled = False
        cancel_button.disabled = True
        return

    # Parse emails: one per line; ignore blank lines; dedupe while preserving order
    raw_lines = emails_text.value.splitlines()
    emails = []
    seen = set()
    for line in raw_lines:
        e = line.strip()
        if not e:
            continue
        if "@" not in e:
            continue
        if e in seen:
            continue
        seen.add(e)
        emails.append(e)

    if not emails:
        with status_out:
            clear_output(wait=True)
            print("No valid emails found. Paste one email per line. Lines without '@' are ignored.")
        start_button.disabled = False
        cancel_button.disabled = True
        return

    total = len(emails)
    progress_bar.max = total
    progress_bar.value = 0
    _stop_flag.clear()

    results = []  # list of dicts

    # Show initial live UI
    with results_out:
        clear_output(wait=True)
        display(progress_bar)
        display(widgets.HTML(f"<b>0 / {total}</b> processed"))
        display(pd.DataFrame(columns=["email","result","deliverable","status","domain","did_you_mean"]))

    # Sequential checks with live updates (safe and easy to read)
    for idx, email in enumerate(emails, start=1):
        if _stop_flag.is_set():
            break
        # Do the call
        try:
            r = verify_email(api_key, email)
        except Exception as e:
            r = {"email": email, "status": "error", "error": str(e)}
        results.append(r)

        # Update progress and live table
        progress_bar.value = idx
        with results_out:
            clear_output(wait=True)
            display(progress_bar)
            display(widgets.HTML(f"<b>{idx} / {total}</b> processed"))
            # Build dataframe for display
            df = pd.DataFrame(results)
            # Keep columns in friendly order if they exist
            cols = [c for c in ["email","result","deliverable","status","domain","did_you_mean","error"] if c in df.columns]
            display(df[cols])
        # Print a short log line
        with status_out:
            clear_output(wait=True)
            last = results[-1]
            if last.get("status") == "success" or last.get("status") == "ok":
                print(f"[{idx}/{total}] {email} → {last.get('result')} (deliverable: {last.get('deliverable')})")
            elif last.get("status") == "error":
                print(f"[{idx}/{total}] {email} → ERROR: {last.get('error') or last.get('raw')}")
            else:
                print(f"[{idx}/{total}] {email} → {last.get('result') or last.get('status')}")
        # small delay to be polite and allow UI refresh
        time.sleep(0.15)

    # Finalize
    final_df = pd.DataFrame(results)
    # Reorder friendly columns if present
    cols_pre = ["email","result","deliverable","status","user","domain","did_you_mean","error"]
    cols = [c for c in cols_pre if c in final_df.columns] + [c for c in final_df.columns if c not in cols_pre]
    final_df = final_df[cols]

    # Save CSV and provide download link
    out_fname = "verimail_results.csv"
    final_df.to_csv(out_fname, index=False)

    with download_link_out:
        clear_output(wait=True)
        print(f"Saved results to `{out_fname}`.")
        display(widgets.HTML(f'<a href="files/{out_fname}" target="_blank" download>Click here to download CSV</a>'))
        # Colab's files.download is more reliable for automatic download:
        try:
            files.download(out_fname)
        except Exception:
            pass

    with status_out:
        print("Verification finished." + (" (cancelled by user)" if _stop_flag.is_set() else ""))

    start_button.disabled = False
    cancel_button.disabled = True

def cancel_verification(_):
    _stop_flag.set()
    with status_out:
        print("Cancel requested — stopping after current request...")

# Wire up buttons
start_button.on_click(run_verification)
cancel_button.on_click(cancel_verification)
cancel_button.disabled = True

# Layout
ui = widgets.VBox([
    widgets.HTML("<h3 style='margin:0'>Verimail v3 — Batch verifier (Colab)</h3>"),
    widgets.HTML("<i>Paste one email per line in the box below. Provide your Verimail API key (hidden) at top.</i>"),
    api_key_widget,
    emails_text,
    widgets.HBox([start_button, cancel_button]),
    widgets.Label("Live status:"),
    status_out,
    widgets.Label("Live results (updates as verifications run):"),
    results_out,
    download_link_out,
    widgets.HTML("<hr><small>Each verification call consumes 1 Verimail credit. Use responsibly.</small>")
])
display(ui)
