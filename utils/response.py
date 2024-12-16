from flask import jsonify
from typing import Optional, Any, Dict, Union

def create_response(
    data: Optional[Dict] = None,
    error: Optional[str] = None,
    status_code: int = 200
) -> Union[Dict[str, Any], tuple]:
    """
    Create a standardized JSON response
    """
    response = {}
    
    if data is not None:
        response.update(data)
    
    if error is not None:
        response['error'] = error
    
    return jsonify(response), status_code