'''
Created on 2022年7月6日

@author: Hsiao-Chien Tsai(蔡効謙)
'''
#from flask import request
from werkzeug.utils import secure_filename
import os 
from nlp import semantic
from utils import file_reader
from utils import filetools
from utils import pptx_tools
UPLOAD_FOLDER = '../web/uploaded_files'
DOWNLOAD_FOLDER = '../web/static/file'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'docx', 'pptx'])

if not os.path.exists(UPLOAD_FOLDER):
  os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(DOWNLOAD_FOLDER):
  os.makedirs(DOWNLOAD_FOLDER)
  
def process_upload(request):
    
    strategy_type = "SWOT"
    # strategy_dict[hit_label][query_text] = corpus_score
    strategy_dict = {"S":{"無":0.0},"W":{"無":0.0},"O":{"無":0.0},"T":{"無":0.0}}
    temp_file = "swot_result.pptx"
    temp_folder = "{}".format(DOWNLOAD_FOLDER)
    temp_path = "{}/{}".format(temp_folder, temp_file)        
    # print(' * received form with', list(request.form.items()))
    # check if the post request has the file part
    if "strategy" in request.form :
        strategy_type = request.form["strategy"]
    
    query_text = ""    
    if "text" in request.form and  len(request.form["text"]) > 1 :
        query_text = request.form["text"]
        # print (request.form["text"])
        strategy_dict = semantic.text_to_label(query_text, strategy_type, top_k=5,semantic_correction=False, demo_mode=True) 
    else :    
        
        for file in request.files.getlist('files'):
            file_ext = file.filename.split('.')[-1].lower()
            if file and file_ext in ALLOWED_EXTENSIONS:
                filename = secure_filename(file.filename)
                #print(file.read())
                file_path=os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path) 
                print ("read file : {}".format(file_path))
                text = ""
                if file_ext == "txt" :
                    query_text = filetools.file_to_text(file_path)
                if file_ext == "pptx" :
                    query_text = file_reader.pptx_to_text(file_path)
                if file_ext == "docx" :
                    query_text = file_reader.docx_to_text(file_path)
                if file_ext == "pdf" :
                    query_text = file_reader.pdf_to_text(file_path)
                    
                strategy_dict = semantic.text_to_label(query_text, strategy_type, top_k=5,semantic_correction=False, demo_mode=True) 
                
                
                print(' * file uploaded', filename)
                #print ("content:{}".format(text))
                
            
                os.remove(file_path) # 移除暫存檔案
            print ("finish")
    
    
    label_desc = semantic.label_to_desc(strategy_type)   
    pptx_tools.save_strategy_to_pptx(strategy_dict, label_desc, pptx_title="{}分析".format(strategy_type) ,out_file_name=temp_path, graph_type = strategy_type )
    
    print (strategy_dict)
    res = {}
    
    res["table"], type_score = semantic.dict_to_table(strategy_dict, strategy_type)  
    res["score"] = type_score 
    
    sentence_len = len(filetools.cut_sentences(query_text, multi_line=False))
    return res, sentence_len