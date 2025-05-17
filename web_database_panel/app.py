from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from config import Config
from models import db, Item
import os
from datetime import datetime

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure the instance folder exists
    os.makedirs(os.path.join(app.instance_path), exist_ok=True)

    # Initialize extensions
    db.init_app(app)

    # Create database tables
    with app.app_context():
        db.create_all()

    # Add template context processors
    @app.context_processor
    def inject_now():
        return {'now': datetime.now()}

    # Register routes
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/items')
    def view_items():
        items = Item.query.all()
        return render_template('view.html', items=items)

    @app.route('/items/add', methods=['GET', 'POST'])
    def add_item():
        if request.method == 'POST':
            name = request.form.get('name')
            description = request.form.get('description')
            category = request.form.get('category')

            if not name:
                flash('Name is required!', 'error')
                return redirect(url_for('add_item'))

            item = Item(name=name, description=description, category=category)
            db.session.add(item)
            db.session.commit()

            flash('Item added successfully!', 'success')
            return redirect(url_for('view_items'))

        return render_template('add.html')

    @app.route('/items/edit/<int:id>', methods=['GET', 'POST'])
    def edit_item(id):
        item = Item.query.get_or_404(id)

        if request.method == 'POST':
            item.name = request.form.get('name')
            item.description = request.form.get('description')
            item.category = request.form.get('category')

            if not item.name:
                flash('Name is required!', 'error')
                return redirect(url_for('edit_item', id=id))

            db.session.commit()
            flash('Item updated successfully!', 'success')
            return redirect(url_for('view_items'))

        return render_template('edit.html', item=item)

    @app.route('/items/delete/<int:id>', methods=['POST'])
    def delete_item(id):
        item = Item.query.get_or_404(id)
        db.session.delete(item)
        db.session.commit()

        flash('Item deleted successfully!', 'success')
        return redirect(url_for('view_items'))

    # API routes for AJAX operations
    @app.route('/api/items', methods=['GET'])
    def api_get_items():
        items = Item.query.all()
        return jsonify([item.to_dict() for item in items])

    @app.route('/api/items/<int:id>', methods=['GET'])
    def api_get_item(id):
        item = Item.query.get_or_404(id)
        return jsonify(item.to_dict())

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)