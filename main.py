import dir.train_model as train_model
from dir.create_csv import create_csv_from_input
from dir.calculate_ratios import calculate_ratios
from dir.compute_metrics import compare_names
from flask import Flask,request,render_template
from flask_cors import CORS
from utils.response import create_response
app=Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/similar_names', methods=['POST'])
def get_name():
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        print("name:",name)
        if not name:
            return create_response(error="name parameter is required", status_code=400)
        output_file = create_csv_from_input(name)
        output_file1 = calculate_ratios(output_file)
        result = compare_names(output_file1)
        return create_response(data={"similar_names": result})
    except Exception as e:
        return create_response(error=str(e), status_code=500)
if __name__=='__main__':
    app.run(debug=True)