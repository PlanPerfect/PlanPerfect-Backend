# PlanPerfect Backend Services Documentation

## Bootcheck

Validates the server environment before startup, ensuring all required configurations and connections are properly set up.

**Usage:** Bootcheck runs on system boot, no additional steps required.

**What it checks:**
- `.env` file existence
- Required environment variables
- Firebase credentials file
- Firebase database connectivity

---

## DatabaseManager

Singleton class for managing Firebase Realtime Database operations with local in-memory caching. Serves as a basic DBMS for the Realtime Database (RTDB). DatabaseManager initialises on system boot.

### `data`
Sets data at a specific path. This method is built-in and does not require parantheses.
```python
DatabaseManager.data["Accounts"]["123"] = newAccount
```

### `peek(path)`
Retrieves data at a specific path without modifying local state.
```python
user_data = DatabaseManager.peek(["Accounts"]["123"])
```

### `destroy(path)`
Deletes data at specified path.
```python
success = DatabaseManager.destroy(["Accounts"]["123"])
```

### `save()`
Saves local data back to Firebase. This method **MUST** be used for changes to persist in RTDB.
```python
DatabaseManager.save()
```

**Sample Usage:**
```python
# Import DatabaseManager
from Services import DatabaseManager as DM

# Read data
user = DM.peek(["Accounts", "123"]) # use list indices separated by commas for class methods

# Modify data
DM.data["Accounts"]["123"] = newAccount # use direct-indexing for data attribute

# Delete data
DM.destroy(["Accounts", "123"]) # use list indices separated by commas for class methods

# Save changes
DM.save()
```

---

## Emailer

Sends HTML emails using Jinja2 templates and SMTP.

**Method:** `send_email(to, subject, template, variables)`

**Parameters:**
- `to` (str): Recipient email address
- `subject` (str): Email subject line
- `template` (str): Template filename without `.html` extension
- `variables` (dict): Variables to render in template

**Sample Usage:**
```python
result = emailer.send_email(
    to="user@example.com",
    subject="Welcome to PlanPerfect!",
    template="welcome", # assuming you have a email template -> /templates/welcome.html
    variables={
        "name": "John Appleseed",
        "activation_link": "https://app.planperfect.com/activate/xyz"
    }
)
```

**Template Example** (`templates/welcome.html`):
```html
<!DOCTYPE html>
<html>
<body>
    <h1>Welcome, {{ name }}!</h1>
    <p>Click <a href="{{ activation_link }}">here</a> to activate your account.</p>
</body>
</html>
```

---

## Logger

Simple file-based logging with timestamps. Logs available at `/logs`

**Usage:**
```python
from Services import Logger

Logger.log("SUCCESS: User logged in.")
Logger.log("UERROR: Invalid inputs.")
Logger.log("ERROR: Database connection failed.")
```

---

## Utilities

Helper functions for common operations.

### `GenerateRandomID(length=None)`
Generates a random UUIDv4 identifier.
```python
from Services import Utilities

id1 = Utilities.GenerateRandomID() # default length is 32
id2 = Utilities.GenerateRandomID(length=8)
```

### `GenerateRandomInt(min_value=0, max_value=100)`
Generates a random integer.
```python
from Services import Utilities

num1 = Utilities.GenerateRandomInt() # default range is 0 - 100
num2 = Utilities.GenerateRandomInt(25, 50)
```

### `HashString(input_string)`
Creates SHA-256 hash of input string.
```python
from Services import Utilities

hashed = Utilities.HashString("myPassword123")
```

### `EncodeToBase64(input_string)`
Encodes string to Base64.
```python
from Services import Utilities

encoded = Utilities.EncodeToBase64("Hello World")
```

### `DecodeFromBase64(encoded_string)`
Decodes Base64 string.
```python
from Services import Utilities

decoded = Utilities.DecodeFromBase64("SGVsbG8gV29ybGQ=")
```

---

**Documentation last updated on** `1 Jan 11.32PM`