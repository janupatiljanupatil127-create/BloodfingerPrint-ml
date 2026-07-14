
"""
History Routes
"""

from flask import Blueprint, jsonify

from utils.database import (
    get_history,
    delete_history
)

history_bp = Blueprint(
    "history",
    __name__
)


@history_bp.route(
    "/history",
    methods=["GET"]
)
def history():

    try:

        data = get_history()

        return jsonify({

            "success": True,

            "count": len(data),

            "history": data

        })

    except Exception as error:

        return jsonify({

            "success": False,

            "message": str(error)

        }), 500


@history_bp.route(
    "/history/<int:record_id>",
    methods=["DELETE"]
)
def remove_history(record_id):

    try:

        delete_history(record_id)

        return jsonify({

            "success": True,

            "message": "History deleted."

        })

    except Exception as error:

        return jsonify({

            "success": False,

            "message": str(error)

        }), 500
