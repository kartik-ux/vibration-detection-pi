# Contributing

Thanks for helping improve Vibration Detection Pi.

## Development setup

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Install the project with `pip install -e '.[test]'`.
4. Run the test suite with `pytest -q`.

## Pull requests

- Keep changes focused and explain the engineering motivation.
- Add or update tests for behaviour changes.
- Document new configuration options and CLI commands.
- Do not commit credentials, private sensor data or machine-identifying information.
- Clearly distinguish simulated results from measurements collected on physical equipment.

## Hardware and dataset contributions

Include the sensor model, sampling rate, mounting arrangement, machine context and data-collection procedure where safe to disclose. Do not publish confidential industrial data without permission.
