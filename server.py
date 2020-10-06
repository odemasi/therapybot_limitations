from __future__ import absolute_import, division, print_function
from flask import Flask, request, jsonify, render_template, session
from flask_socketio import SocketIO, emit, join_room

import json
import sqlite3
from datetime import date
import datetime
import time
import torch
import pickle as pkl
import random
import sys

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)



# GPU = 6
# SANDBOX = True
# DEBUG = True

NUM_WRITTEN_AT_START = 0
# if DEBUG and SANDBOX: 
#     NUM_WRITTEN_AT_START = 10
#     SUBMIT=False
# else:
#     NUM_WRITTEN_AT_START = 0
#     SUBMIT=True
#     
#     
#     
# if SANDBOX: 
#     if SUBMIT:
#         turk_submit_url = 'https://workersandbox.mturk.com/mturk/externalSubmit'
#     else: 
#         turk_submit_url = ''
# else:
#     turk_submit_url = "https://www.mturk.com/mturk/externalSubmit"
    


# CONDITION_MESSAGES = {'1': ['condition 1: response 1', 'condition 1: response 2', 'condition 1: response 3'],
#                     '2': ['condition 2: response 1', 'condition 2: response 2', 'condition 2: response 3']}
CONDITION_MESSAGES = {'%s' % x: ['condition %s: response %s' % (x, y) for y in range(11)] for x in range(2)}
CONDITION_NAMES = list(CONDITION_MESSAGES.keys())
    

clients = []
session_info = {}

@app.route('/')
def root():
    """ Send HTML from the server."""
    return render_template('index.html')
    
    
# @app.route('/counselor/', methods=['GET'])
# def counselor_root():
#     """ Send HTML from the server."""
#     return render_template('counselor_index.html')
    
    
    
    
# @app.route('/covidbot/counselor_distinct/', methods=['GET'])
# def counselor_distinct_root():
#     """ Send HTML from the server."""
#     return render_template('counselor_distinct.html')
    
    
    
    
@app.route('/user/', methods=['GET'])
def user_root():
    """ Send HTML from the server."""
    return render_template('user.html')
    
    
    
    
# @app.route('/covidbot/mturk/', methods=['GET'])
# def mturk_root():
# 
#     """ Send HTML from the server."""
#     assignmentId = request.args.get('assignmentId')
#     hitId = request.args.get('hitId')
#     workerId = request.args.get('workerId')
#     
#     return render_template('mturk_index.html', turk_submit_url=turk_submit_url, assignmentId=assignmentId, hitId=hitId, workerId=workerId)
    
    
    
# @app.route('/covidbot/test/', methods=['GET'])
# def test_root():
#     """ Send HTML from the server."""
#     return render_template('test_index.html')

    
    

# @socketio.on('joined')
# def joined(message):
#     """Sent by clients when they enter a room.
#     A status message is broadcast to all people in the room."""
#     
# #     assignmentid = message['assignmentid']
# #     hitid = message['hitid']
#     condition = random.choice(CONDITIONS)
#     # Add client to client list
#     session_info[request.sid] = {'joined_time':datetime.datetime.now(), 
#                                     'num_written': NUM_WRITTEN_AT_START,
#                                     'condition': condition
# #                                     'assignmentid': assignmentid,
# #                                     'hitid': hitid, 
# #                                     'model_name': MTURK_MODEL_NAMES[random.randrange(len(MTURK_MODEL_NAMES))]
#                                     } 
#     
#     clients.append(request.sid)
# 
#     room = session.get('room')
#     join_room(room)
#     
#     first_message = 'Hello, thanks for joining the study!' 
# #     with sqlite3.connect('data/session_info.db') as conn:
# #         cur = conn.cursor()
# #         cur.executemany("INSERT INTO message_pairs VALUES (?, ?, ?, ?, ?)", [(str(request.sid), 'START', first_message, assignmentid, hitid),])
# #         conn.commit()
#         
#     
#     emit('render_sys_message', {"data": first_message}, room=request.sid) 
    
    
    
    
@socketio.on('user_joined')
def user_joined(message):
    """Sent by clients when a user enters a room.
    A status message is broadcast to all people in the room."""

    condition = random.choice(CONDITION_NAMES)
    # Add client to client list
    session_info[request.sid] = {'joined_time':datetime.datetime.now(), 
                                    'num_written': NUM_WRITTEN_AT_START,
                                    'condition': condition,
#                                     'assignmentid': assignmentid,
#                                     'hitid': hitid, 
#                                     'model_name': MTURK_MODEL_NAMES[random.randrange(len(MTURK_MODEL_NAMES))]
                                    'convo': []} 
    
    clients.append(request.sid)

    room = session.get('room')
    join_room(room)
    
#     first_message = 'Hello, thanks for joining the study!' 
    
    
    emit('user_render_sys_message', {"data": '[Please enter your participant ID above. You may start after the system sends the first message.]'}, room=request.sid) 
        
    
    
    
def get_previous_interactions(participantid):

    with sqlite3.connect('data/session_info.db') as conn:
        cur = conn.cursor()
        results = cur.execute('select conditionfrom user_feedback where participantid="%s"' % participantid).fetchall()
        
        condition_names = list(set([x[0] for x in results]))
        
    return condition_names
    
    

def get_condition_for_participant(participantid):   
    selected_condition = random.choice(CONDITION_NAMES) 
    
#  DEBUG: CHOOSE THE UNSEEN CONDITIONS:
#     prev_condition_names = get_previous_interactions(participantid)
#     unseen_conditions = [x for x in CONDITION_NAMES if x not in prev_condition_names]
#     
#     
#     if len(unseen_models) > 0:
#         # sample a new model from unseen
#         selected_condition = unseen_conditions[random.randrange(len(unseen_conditions))]
#     else:
#         # all models have been seen, so sample from full list again.
#         selected_condition = CONDITION_NAMES[random.randrange(len(CONDITION_NAMES))]
    
    return selected_condition




@socketio.on('user_sent_pid')
def user_sent_pid(message):
    # resent hitid to the counselor's participant id
#     print('message: ', message)
#     locations = ['top', 'middle', 'bottom'] if message["version"] == 'distinct' else ['left', 'center', 'right']
    
    participantid = message["data"]
    selected_condition = get_condition_for_participant(participantid)
#     print(selected_model, personaids)
    
    emit('render_pid', {"data":'Participant ID "%s". Session loading first message' %  message["data"]}, room=request.sid)
        
#         with sqlite3.connect('data/session_info.db') as conn:
#             cur = conn.cursor()
#             db_input = (compound_sid, 'START', first_message, participantid, session_info[compound_sid]['location_model_name'])
#             cur.executemany("INSERT INTO counselor_message_pairs VALUES (?, ?, ?, ?, ?)", [db_input,])
#             conn.commit()
# #         print('INTO MESSAGE_PAIRS: ', db_input)
# #         print(location, 'Persona: ', personaids[l])
#         
    first_message = CONDITION_MESSAGES[selected_condition][0]
    session_info[request.sid]['convo'].append(('START', first_message))
    
    emit('render_sys_message', {"data": first_message }, room=request.sid) 
    
    print("participant id entered")
    







@socketio.on('user_sent_message')   
def user_sent_message(message):
    """
    Called when the counselor sends a message.
    """
    
    if len( message["data"].strip()) > 0:
        session_info[request.sid]['num_written'] += 1
    message['count'] = session_info[request.sid]['num_written']
    emit('render_usr_message', message, room=request.sid)
#   print(request.sid)
    
    selected_condition = session_info[request.sid]['condition']
    
    # Extract a string of the user's message
    raw_user_input_text = message["data"]
    if len(raw_user_input_text.strip()) > 0:
        
#         input_text = raw_user_input_text.lower() # debug: review the preprocessing of the raw input text.
        
#         output_text, response_info = MODEL_DICT[selected_model].chat(input_text, 
#                                                                     compound_sid, 
#                                                                     '', # assignmentid
#                                                                     '') # hitid
        print(CONDITION_MESSAGES[selected_condition])
        print(session_info[request.sid]['num_written'])
        output_text = CONDITION_MESSAGES[selected_condition][session_info[request.sid]['num_written']]                                        

# DEBUG: STORE MESSAGES        
#         with sqlite3.connect('data/session_info.db') as conn:
#             cur = conn.cursor()
#             db_input = (compound_sid, input_text, output_text, 
#                         session_info[request.sid]['participantid'],    
#                         session_info[request.sid]['condition'])
#             cur.executemany("INSERT INTO counselor_message_pairs VALUES (?, ?, ?, ?, ?)", [db_input])
#             conn.commit()
        
#         print('Into counselor_message_pairs: ', db_input)
        
#         output_text = output_text.replace('fucking', '<EXPLETIVE>').replace('fuck', '<EXPLETIVE>')
    
        # Render our response
        emit('render_sys_message', {"data": output_text}, room=request.sid)
#         response_info_str = json.dumps(response_info) # debug: uncomment storing to sql below.
#         with sqlite3.connect('data/session_info.db') as conn:
#             cur = conn.cursor()
#             cur.executemany("INSERT INTO response_info VALUES (?, ?)", [(compound_sid, response_info_str,)])
#             conn.commit()
        
        session_info[request.sid]['convo'].append((raw_user_input_text, output_text))
        
        
    else:
        emit('render_sys_message', {"data": '[oops, please enter message text]'}, room=request.sid)
        
        
        
#         
# @socketio.on('user_sent_message')   
# def user_sent_message(message):
#     """
#     Called when the user sends a message.
#     """
#     # This renders the user message we have received on screen
#     if len( message["data"].strip()) > 0:
#         session_info[request.sid]['num_written'] += 1
#     message['count'] = session_info[request.sid]['num_written']
#     emit('render_usr_message', message, room=request.sid)
# #   print(request.sid)
#     
#     
#     # Extract a string of the user's message
#     raw_user_input_text = message["data"]
#     if len(raw_user_input_text.strip()) > 0:
#         
#         input_text = raw_user_input_text.lower() # debug: review the preprocessing of the raw input text.
#         
#         # output_text, sys_da_output, sys_se_output, usr_da_output, usr_se_outpu,candidates = model.chat(input_text, request.sid)
# 
# #         debug: remove this, uncomment model below.
# #         output_text = 'blah' 
# #         response_info = {'test': [0]}
# #         print('PERSONA ID: ', model.session_info[request.sid]['personality_id'])
# #         print('FIRST 2 PERSONA SEGMENTS: ', model.session_info[request.sid]['persona_segments'][:2])
#         model_name = session_info[request.sid]['model_name']
#         output_text, response_info = MODEL_DICT[model_name].chat(input_text, 
#                                                 request.sid, 
#                                                 session_info[request.sid]['assignmentid'],    
#                                                 session_info[request.sid]['hitid'])
#                                                 
#         with sqlite3.connect('data/session_info.db') as conn:
#             cur = conn.cursor()
#             cur.executemany("INSERT INTO message_pairs VALUES (?, ?, ?, ?, ?)", 
#                                             [(str(request.sid), input_text, output_text, 
#                                             session_info[request.sid]['assignmentid'],    
#                                             session_info[request.sid]['hitid'])])
#             conn.commit()
#         output_text = output_text.replace('fucking', '<EXPLETIVE>').replace('fuck', '<EXPLETIVE>')
#     
#     
#         # Render our response
#         emit('render_sys_message', {"data": output_text}, room=request.sid)
#         response_info_str = json.dumps(response_info) # debug: uncomment storing to sql below.
#         with sqlite3.connect('data/session_info.db') as conn:
#             cur = conn.cursor()
#             cur.executemany("INSERT INTO response_info VALUES (?, ?)", [(str(request.sid), response_info_str,)])
#             conn.commit()
#         
#         if 'convo' in session_info[request.sid].keys():
#             session_info[request.sid]['convo'].append((input_text, output_text))
#         
#         
#     else:
#         emit('render_sys_message', {"data": '[oops, please enter message text]'}, room=request.sid)
#     
#     
# 
#     # Render dialog act
# #   emit('start_da', {}, room=request.sid)
# 
# #   emit('start_candidate', {}, room=request.sid)
# #   for i, candidate in enumerate(candidates):
# #       emit('render_candidate', {"data": candidate[0], "sysda": candidate[1]}, room=request.sid)
# 
# 
# 
# @socketio.on('user_sent_feedback')  
# def user_sent_feedback(feedback):
#     """
#     Called when the user sends feedback.
#     """
#     # Extract a string of the users message
#     feedback_text = feedback["data"]
#     with sqlite3.connect('data/session_info.db') as conn:
#         cur = conn.cursor()
#         cur.executemany("INSERT INTO feedback VALUES (?, ?)", [(str(request.sid), feedback_text),])
#         conn.commit()
#         
#     emit('render_feedback', {"data":'Feedback received. Thanks!'}, room=request.sid)
# 
# 
# @socketio.on('log_mturk_feedback') 
# def log_mturk_feedback(feedback):
#     
#     with sqlite3.connect('data/session_info.db') as conn:
#         cur = conn.cursor()
#         duration = (datetime.datetime.now() - session_info[request.sid]['joined_time']).total_seconds()
#         submit_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         joined_time = session_info[request.sid]['joined_time'].strftime("%Y-%m-%d %H:%M:%S")
#         
#         model_name = session_info[request.sid]['model_name']
#         pid = MODEL_DICT[model_name].session_info[request.sid]['personality_id']
#         sid = request.sid
#         num_written = session_info[request.sid]['num_written']
#         
#         db_input = (sid, joined_time, submit_time, duration, model_name, pid, num_written, str(SANDBOX), feedback["assignmentId"], feedback["hitId"], feedback["workerId"], feedback['data'])
#         cur.executemany("INSERT INTO mturk_feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [db_input,])
#         conn.commit()
#     
# #     print(db_input) # debug: comment
#     MODEL_DICT[model_name].clear_session_info(request.sid) # debug: uncomment
#     
#     
    
    
@socketio.on('log_user_feedback') 
def log_user_feedback(feedback):
    
    print('NEED TO ADD FEEDBACK LOGGING')
    
#     with sqlite3.connect('data/session_info.db') as conn:
#         cur = conn.cursor()
#         duration = (datetime.datetime.now() - session_info[compound_sid]['joined_time']).total_seconds()
#         submit_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         joined_time = session_info[compound_sid]['joined_time'].strftime("%Y-%m-%d %H:%M:%S")
#         
#         selected_model = session_info[compound_sid]['location_model_name']
#         pid = MODEL_DICT[selected_model].session_info[compound_sid]['personality_id']
#         num_written = session_info[compound_sid]['num_written']
#         location_model_name = session_info[compound_sid]['location_model_name']
#         participantid = session_info[compound_sid]['participantid'] # this is set as the participant ID in the counselor join function.
#         
#         
#         db_input = (compound_sid, joined_time, submit_time, duration, location_model_name, pid, num_written, str(SANDBOX), participantid, feedback['data'])
#         cur.executemany("INSERT INTO counselor_feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [db_input,])
#         conn.commit()
# #     print('added to db: ', db_input)
#     
#     pickle_name = 'counselor_logs/feedback_counselor_%s_sid_%s_joined_%s.pkl' % (participantid, compound_sid, submit_time)
#     pickle_name = pickle_name.replace(' ', '_')
#     pkl.dump((compound_sid, session_info[compound_sid], db_input), open(pickle_name, 'wb'))   
# #     print('pickle dumped:')
# #     print((compound_sid, session_info[compound_sid], db_input))
#     
#     
# #     print(db_input) # debug: comment
#     if compound_sid in MODEL_DICT[selected_model].session_info.keys():
#         MODEL_DICT[selected_model].clear_session_info(compound_sid) # debug: uncomment
    
    
    
    

        
        
if __name__ == '__main__':
    """ Run the app. """
    socketio.run(app, host='0.0.0.0', port=6776)
    
    
#     CUDA_VISIBLE_DEVICES=$GPU gunicorn -k gevent --timeout 60 -w 1 -b 127.0.0.1:6776 server:app;
# Yu: to fix session timeout? https://stackoverflow.com/questions/32390268/socket-io-how-to-change-timeout-value-option-on-client-side
# print(torch.cuda.is_available())
# run: gunicorn --worker-class eventlet -w 1 -b 127.0.0.1:6776 server:app
# navigate to: https://language.cs.ucdavis.edu/covidbot/user/

