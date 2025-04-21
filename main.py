import dir.train_model as train_model
from dir.create_csv import create_csv_from_input
from dir.calculate_ratios import calculate_ratios
from dir.compute_metrics import compare_names
from flask import Flask,request,render_template
from flask_cors import CORS
from utils.response import create_response
app=Flask(__name__)
CORS(app)
from indic_transliteration import sanscript
from langdetect import detect
def detect_language(name):
    try:
        # Detect the language/script of the input name
        lang_code = detect(name)
        return lang_code
    except Exception as e:
        return None
def convert_to_english(name):
    script=''
    lang_code = detect_language(name)
    if lang_code=='en':
       return name,script,False
    else:
        supported_scripts = {
        'hi': sanscript.DEVANAGARI,  # Hindi
        'kn': sanscript.KANNADA,    # Kannada
        'te': sanscript.TELUGU,     # Telugu
        'ta': sanscript.TAMIL,      # Tamil
        'ml': sanscript.MALAYALAM,  # Malayalam
        'gu': sanscript.GUJARATI,   # Gujarati
        'pa': sanscript.GURMUKHI    # Punjabi (Gurmukhi)
       }
        if lang_code in supported_scripts:
            script = supported_scripts[lang_code]
            english_name = sanscript.transliterate(name, script, sanscript.ITRANS)
            return english_name,script,True

def convert_to_lang(name,script):
    name=sanscript.transliterate(name,sanscript.ITRANS,script)
    return name

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/similar_names', methods=['POST'])
def get_name():
    # try:
    data = request.get_json()
    name = data.get('name', '').strip()
    name,script,bool = convert_to_english(name)
    print("name:", name)
    if not name:
        return create_response(error="name parameter is required", status_code=400)
    output_file = create_csv_from_input(name)
    output_file1 = calculate_ratios(output_file)
    result = compare_names(output_file1)
    if bool:
        result = [convert_to_lang(name,script) for name in result]
    return create_response(data={"similar_names": result})
    # except Exception as e:
    #     return create_response(error=str(e), status_code=500)
if __name__=='__main__':
    app.run(debug=True)