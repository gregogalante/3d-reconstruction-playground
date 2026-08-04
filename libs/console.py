"""Coloured console output shared by the pipeline and its subprocesses."""

def print_error(message):
  color = "\033[91m"  # Red color
  reset = "\033[0m"
  print(f"{color}ERROR: {message}{reset}")

def print_success(message):
  color = "\033[92m"  # Green color
  reset = "\033[0m"
  print(f"{color}SUCCESS: {message}{reset}")

def print_info(message):
  color = "\033[96m"  # Cyan color
  reset = "\033[0m"
  print(f"{color}INFO: {message}{reset}")

def print_warning(message):
  color = "\033[93m"  # Yellow color
  reset = "\033[0m"
  print(f"{color}WARNING: {message}{reset}")

def print_step(message):
  color = "\033[94m"  # Blue color
  reset = "\033[0m"
  print("\n")
  print(f"{color}{'='*100}{reset}")
  print(f"{color}RUNNING STEP: {message}{reset}")
  print(f"{color}{'='*100}{reset}\n")
