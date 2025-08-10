# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Station de Réappropriation is a centralized registry of autonomous tools for managing electricity charging stations and billing processes. It serves as a libre complement to Odoo ERP, focusing on French energy market compliance and reducing dependencies on third-party capitalist tools.

**Core Domain**: Electricity station management, ENEDIS data processing, TURPE billing calculations, and consumption analysis for French energy distribution networks.

## Development Environment

### Dependencies and Setup
- **Python**: 3.12.x (strict requirement)
- **Package Manager**: Poetry
- **Main Framework**: Marimo for interactive notebooks
- **External Dependencies**: 
  - `electricore` (local dependency at ../electricore, develop mode)
  - `electriflux` for ENEDIS data processing
  - Odoo XML-RPC for ERP integration

### Common Commands

```bash
# Install dependencies
poetry install

# Setup autostart service (system-wide)
poetry run stationreappropriation-setup-autostart
# or via script alias
sudo setup-marimo-autostart

# Remove autostart service
poetry run stationreappropriation-remove-autostart
```

## Architecture

### Core Modules

**`utils.py`**: Central utilities for date generation, environment configuration, and consumption data handling
- Date functions: `gen_dates()`, `gen_trimester_dates()`, `gen_month_boundaries()`
- Environment: `load_prefixed_dotenv()` for configuration management
- Constants: `get_consumption_names()` returns ['HPH', 'HPB', 'HCH', 'HCB', 'HP', 'HC', 'BASE']

**`odoo/odoo_connector.py`**: Odoo ERP integration via XML-RPC
- Context manager pattern for connection handling
- Simulation mode for testing (`sim=True`)
- CRUD operations with automatic type conversion for numpy/pandas data

**`marimo_utils.py`**: SFTP download utilities with Marimo progress bars
- Encrypted ENEDIS data download from FTP servers
- Progress tracking for large data processing tasks

### Data Processing Architecture

**ENEDIS Data Flows**:
- **C15**: Contractual events (activations, disconnections, supplier changes)
- **R151**: Daily consumption readings  
- **F15/F12**: Billing data for different customer categories
- **R63**: Time-series consumption data with PRM identifiers

**Processing Pipeline**:
1. SFTP download and decryption (`marimo_utils.py`)
2. Data parsing and validation (`experimentations/`)
3. Billing calculations with TURPE compliance
4. Odoo ERP integration for invoicing

### Configuration Management

Environment variables are loaded with prefixes:
- `SR_*`: Station Reappropriation general config
- `EOB_*`: Default prefix for energy/billing operations
- Required variables typically include: `ODOO_URL`, `ODOO_DB`, `ODOO_USERNAME`, `ODOO_PASSWORD`
- SFTP variables: `FTP_ADDRESS`, `FTP_USER`, `FTP_PASSWORD`, `AES_KEY`, `AES_IV`

Configuration files expected at: `~/station_reappropriation/.env`

## Key Experiments and Tools

**Consumption Analysis** (`experimentations/analyse_charge.py`):
- Statistical decomposition of consumption patterns
- Weather correlation analysis using OpenMeteo API
- Time-series visualization and monitoring

**Billing Pipeline** (`experimentations/demo.py`):
- Complete automation of ENEDIS data to Odoo invoicing
- TURPE fixed costs and CTA calculations
- Multi-flux data integration

**Period Management** (`experimentations/periodes.py`):
- Billing period detection from contractual events
- Subscription lifecycle management
- Complex tariff modification handling

**Tariff Modeling** (`experimentations/tarifs.py`):
- Pricing structure and revenue projections
- Break-even analysis for subscriber growth
- Fixed charge modeling (€3.31 operations + network fees + taxes)

## Development Guidelines

### Marimo Best Practices

**UI Component Separation**: Always separate UI widget creation from their usage in different cells:
```python
# Cell 1: Create UI widgets
@app.cell
def _():
    folder_picker = mo.ui.file_browser(selection_mode="directory", label="Select folder")
    return (folder_picker,)

# Cell 2: Use UI widgets  
@app.cell
def _(folder_picker):
    selected_path = folder_picker.value
    return selected_path,
```

**File/Folder Selection**: Use `mo.ui.file_browser()` with appropriate `selection_mode` parameter:
- `selection_mode="file"` for single file selection
- `selection_mode="directory"` for folder selection

### Code Conventions
- French language for business domain terms and comments
- Use context managers for external connections (Odoo, SFTP)
- Pandas DataFrame operations preferred for data manipulation
- Type hints required for function signatures
- Use `pathlib.Path` for file operations

### Data Handling
- PRM (Point Référence Mesure) identifiers for meter identification
- Date handling follows French billing cycles (monthly periods)
- Consumption data in HPH/HPB/HCH/HCB format (French tariff periods)
- Currency calculations in euros with proper precision

### Testing and Validation
- Use simulation mode (`sim=True`) in OdooConnector for testing
- Validate required configuration parameters before processing
- Handle ENEDIS data format variations gracefully
- Progress bars required for long-running operations

### External Integrations
- ENEDIS SFTP servers for official energy data
- Odoo ERP via XML-RPC (models: sale.order, account.move, etc.)
- OpenMeteo API for weather correlation analysis
- Paramiko for secure SFTP operations with encryption