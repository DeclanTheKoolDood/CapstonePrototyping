# Dark Theme Database Panel

A sleek, dark-themed Flask application for interfacing with a SQLite database. This application provides a modern UI for performing CRUD operations on your database.

## Features

- 🌙 Dark theme UI with modern aesthetics
- 📊 View, add, edit, and delete database records
- 🔍 Search and filter functionality
- 🔄 Sorting capabilities
- 📱 Responsive design for all devices
- 🔌 REST API endpoints for programmatic access

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python app.py
   ```
2. Open your browser and navigate to `http://127.0.0.1:5000`

## Database Structure

The application uses SQLite with SQLAlchemy ORM. The default model includes:

- `Item`: A basic model with name, description, category, and timestamps

You can extend the models in `models.py` to fit your specific needs.

## API Endpoints

- `GET /api/items`: Get all items
- `GET /api/items/<id>`: Get a specific item by ID

## Customization

- Modify `models.py` to change the database structure
- Edit templates in the `templates` folder to customize the UI
- Adjust styles in `static/css/style.css` to change the appearance

## Technologies Used

- Flask: Web framework
- SQLAlchemy: ORM for database operations
- Bootstrap 5: Front-end framework with dark theme
- Font Awesome: Icons
- JavaScript: Client-side interactivity

## License

MIT
