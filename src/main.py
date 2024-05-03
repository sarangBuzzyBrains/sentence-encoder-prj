from flask import Flask, request
from tensor_service.sentence_encoder import run_and_plot
import json

app = Flask(__name__)



@app.route('/sentence_encoding')
def hello_world():
    if request.is_json:
        json_data = request.get_json()
        message1  =json_data["message1"]
        message2  =json_data["message2"]
        threshold = json_data["threshold"]
        res_matrix = run_and_plot(message1, message2)
              
        filtered_matrix = {}
        ar_idx = 0
        for inp_text in message1:
            filtered_matrix[inp_text] = []
            text_idx = 0
            for rel_val in res_matrix[ar_idx]:
                if(rel_val >= threshold and text_idx != ar_idx):
                    filtered_matrix[inp_text].append(message2[text_idx])
                text_idx += 1
            ar_idx += 1

        final_output = {
            "filtered_data": filtered_matrix,
            "raw_data": convert_num_to_matrix(res_matrix)
        }

        return final_output

    return 'Error'


def convert_num_to_matrix(a):
    b = []
    for i in a:
        arr_list = i.tolist()
        json_data = json.dumps(arr_list)
        b.append(json_data)

    return b


if __name__ == '__main__':
    app.run()