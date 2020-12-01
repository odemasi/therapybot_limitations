
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
import pandas as pd
import copy

import torch

from parlai.core.params import ParlaiParser, print_announcements
from parlai.core.agents import create_agent, create_agent_from_model_file, create_agent_from_opt_file
from parlai.core.logs import TensorboardLogger
from parlai.core.metrics import (
    aggregate_named_reports,
    aggregate_unnamed_reports,
    Metric,
)
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger, nice_report
from parlai.utils.world_logging import WorldLogger
from parlai.core.script import ParlaiScript, register_script
from parlai.utils.io import PathManager
import parlai.utils.logging as logging

from parlai.core.opt import Opt
from parlai.utils.strings import normalize_reply


# For Blender
import requests
import redis
import hashlib



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
# CONDITION_MESSAGES = {'%s' % x: ['condition %s: response %s' % (x, y) for y in range(11)] for x in range(2)}
# CONDITION_NAMES = list(CONDITION_MESSAGES.keys())

chat_flows = pd.read_csv('data/chat_flows.csv')
CONDITION_NAMES = list(chat_flows.columns)
CONDITION_MESSAGES = {x: list(chat_flows[x].values) for x in CONDITION_NAMES}


chat_flows_no_ack = pd.read_csv('data/chat_flows_no_ack.csv')
CONDITION_NAMES_NO_ACK = list(chat_flows_no_ack.columns)
CONDITION_MESSAGES_NO_ACK = {x: list(chat_flows_no_ack[x].values) for x in CONDITION_NAMES_NO_ACK}

print('Flow conditions: ')
print(CONDITION_NAMES)

# parser = ParlaiParser(True, True, 'Evaluate a model')
# opt = parser.parse_args()
# agent = create_agent(opt, requireModelExists=True)
THERAPYBOT_GPU = 1 # These index the visible devices?
REDDIT_GPU = 0


##### overfit on full crisisbot counselor messages
# opt = Opt(
#             {
#                 'datapath': 'dummy_path',
#                 'model': 'hugging_face/gpt2',
#                 'init_model': None,
#                 'model_file': '/home/oademasi/transfer-learning-conv-ai/ParlAI/trained_models/therapybot_v1_gpt2_small',
# #                 'model': 'image_seq2seq',
# #                 'init_model': None,
# #                 'model_file': '/home/oademasi/transfer-learning-conv-ai/ParlAI/trained_models/fine_tuned_base_dodeca',
#                 'load_from_checkpoint': True,
#                 'override':{'gpu': THERAPYBOT_GPU, 'inference': 'beam', 'beam_size': 5, 'skip_generation': False, 'beam_block_ngram': 3, 'beam_context_block_ngram': 3}
#             }
#         )
        

#### Crisisbot first counselor sentence + empathetic dialogues weighted 1:9
opt = Opt(
            {
                'datapath': 'dummy_path',
                'model': 'hugging_face/gpt2',
                'init_model': None,
#                 'model_file': '/home/oademasi/transfer-learning-conv-ai/ParlAI/trained_models/therapybot_v3_first_sent_gpt2_small',
                'model_file': '/home/oademasi/transfer-learning-conv-ai/ParlAI/trained_models/therapybot_v2_first_sent_gpt2_small',
#                 'model': 'image_seq2seq',
#                 'init_model': None,
#                 'model_file': '/home/oademasi/transfer-learning-conv-ai/ParlAI/trained_models/fine_tuned_base_dodeca',
                'load_from_checkpoint': True,
                'override':{'gpu': THERAPYBOT_GPU, 'beam-min-length': 3, 'inference': 'beam', 'beam_size': 3, 'skip_generation': False, 'beam_block_ngram': 3, 'beam_context_block_ngram': 3, 'beam_block_list_filename': '/home/oademasi/therapybot_limitations/block_list.txt'}
            }
        )        

# agent = create_agent_from_opt_file(opt)
# agent.set_interactive_mode(True)



# reddit_opt = Opt(
#             {
#                 'datapath': 'dummy_path',
#                 'model': 'image_seq2seq',
#                 'init_model': None,
#                 'model_file': '/home/oademasi/transfer-learning-conv-ai/ParlAI/data/models/dodecadialogue/reddit_ft/model',
#                 'load_from_checkpoint': True,
#                 'override':{'gpu': REDDIT_GPU, 
#                             'inference': 'beam', 
#                             'beam_size': 5, 
#                             'skip_generation': False, 
#                             'beam_block_ngram': 3, 
#                             'beam_context_block_ngram': 3, 
# #                             "allow_missing_init_opts": False, 
#                             'parlai_home': '/home/oademasi/transfer-learning-conv-ai/ParlAI', 
#                             "image_features_dim": 2048,
#                             "image_encoder_num_layers": 1,
#                             "n_image_tokens": 1,
#                             "n_image_channels": 1,
#                             "include_image_token": True,
#                             "image_fusion_type": "late",
#                             'beam_min_length': 10
#                             }
#             }
#         )
# reddit_agent = create_agent_from_opt_file(reddit_opt)
# reddit_agent.set_interactive_mode(True)

# agents = {'therapybot': agent, 'ethics_base': reddit_agent}

# agents = {'therapybot': agent}
USE_BLENDER = False


# agent.opt['inference'] = 'beam'
# agent.opt['beam_size'] = 5
# # agent.opt['beam_min_length'] = 10 
# agent.opt['beam_block_ngram'] =  3

print('parlai reddit agent loaded!')

clients = []
session_info = {}

###################
##### BLENDER #####
###################
BLENDER_ACKNOWLEDGEMENT_URL = "http://54.227.176.241:5007/blender/redis_generate"


def get_blender_response(last_user_text, last_bot_response, current_user_text, key=str(time.time())):
    request_data = {
        "last_turn_text": last_user_text,
        "last_turn_response": last_bot_response,
        "text": current_user_text,
        "blender_id": key
    }

    try:
        requests.post(url=BLENDER_ACKNOWLEDGEMENT_URL, json=request_data, timeout=(1, 0.001))
    except Exception as e:
        pass
        print(e) # exception doesn't matter here, a read timed out is expected here

    time.sleep(1.8)
    result = RedisHelper().get(RedisHelper.PREFIX_BLENDER_ACKNOWLEDGEMENT, key)
    return result

class RedisHelper:
    SERVER_IP = '54.145.233.163'
    PREFIX_BLENDER_ACKNOWLEDGEMENT = "gunrock:blender:question_handling"

    def __init__(self):
        self.r = redis.StrictRedis(
            host=self.SERVER_IP,
            port=16517,
            db=0,
            password="alexaprize",
            charset='utf-8',
            decode_responses=True)
        
    def get(self, prefix, key):
        key = hashlib.md5(key.encode()).hexdigest()

        results = self.r.get(prefix + ':' + key)
        if results:
            return json.loads(results)
        return None






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
    

@app.route('/annotation/', methods=['GET'])
def annotation_root():
    """ Send HTML from the server."""
    participantId = request.args.get('userId')
    
    return render_template('annotation.html', participantId=participantId)
    
        
    
@app.route('/generation/', methods=['GET'])
def generation_root():
    """ Send HTML from the server."""
    return render_template('generation.html')
    
    
    
@app.route('/interleave/', methods=['GET'])
def interleave_root():
    """ Send HTML from the server."""
    return render_template('interleave.html')
    
    
@app.route('/ethicsbot_a/', methods=['GET'])
def ethicsbot_a_root():
    """ Send HTML from the server."""
    return render_template('ethicsbot_a.html')
    
    
    
    
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
    
    
    
@socketio.on('user_joined_annotation')
def user_joined_annotation(message):
    
    clients.append(request.sid)
    
    pid = message['participantid']
    room = session.get('room')
    join_room(room)
    print(type(pid), pid)
    if pid != 'None':
    	
        with sqlite3.connect('data/session_info.db') as conn:
            cur = conn.cursor()
        
            # get all the relevant sessions for a user
            session_ids = cur.execute('select sid, count(*) as c from message_pairs where pid="%s" group by sid having c > 0'%pid).fetchall()
            if len(session_ids) == 0:
                print('no user sessions?!')
            else:
                sid = session_ids[-1][0] # debug: use the last logged session.
            
    #             message_pairs: sid text, pid text, condition text, message text, response text, num_written
            convo = cur.execute('select message, response from message_pairs where sid="%s" order by exchange_num' % sid).fetchall()
        
            if len(convo) == 0:
                convo = [[pid,'null'],]
        
        emit('render_convo', convo, room=request.sid)
    
    
    

    
@socketio.on('user_joined')
def user_joined(message):
    """Sent by clients when a user enters a room.
    A status message is broadcast to all people in the room."""
    print(message)
#     condition = random.choice(CONDITION_NAMES)
    # Add client to client list
    
    
    if USE_BLENDER:
        parlai_history = ''
        
    else:
        if message['agent'] != 'covid_flow':
            torch.cuda.set_device(agents[message['agent']].opt['gpu'])
        parlai_history = copy.deepcopy(agents[message['agent']].build_history()) if message['agent'] != 'covid_flow' else ''
        
    
    session_info[request.sid] = {'joined_time':datetime.datetime.now(), 
                                    'num_written': NUM_WRITTEN_AT_START,
#                                     'condition': '',
#                                     'assignmentid': assignmentid,
#                                     'hitid': hitid, 
#                                     'model_name': MTURK_MODEL_NAMES[random.randrange(len(MTURK_MODEL_NAMES))]
                                    'convo': [],
                                    'parlai_history': parlai_history} #NOTE: this should build an agent-agnostic empty history, but check the agent doesn't matter.
    
    clients.append(request.sid)

    room = session.get('room')
    join_room(room)
    
#     first_message = 'Hello, thanks for joining the study!' 
    
    
    emit('render_sys_message', {"data": '[Please enter your participant ID above. You may start after the system sends the first message.]'}, room=request.sid) 
        
    
    
    
def get_previous_interactions(pid):

    with sqlite3.connect('data/session_info.db') as conn:
        cur = conn.cursor()
        results = cur.execute('select distinct condition from message_pairs where pid="%s"' % pid).fetchall()
        
        seen_condition_names = [x[0] for x in results]
        
    return seen_condition_names
    
    

def get_condition_for_participant(pid):   
#     selected_condition = random.choice(CONDITION_NAMES) 
    
#  DEBUG: CHOOSE THE UNSEEN CONDITIONS:
    seen_condition_names = get_previous_interactions(pid)
    unseen_conditions = [x for x in CONDITION_NAMES if x not in seen_condition_names]
    
    
    if len(unseen_conditions) > 0:
        # sample a new model from unseen
        selected_condition = random.choice(unseen_conditions)
    else:
        # all models have been seen, so sample from full list again.
        selected_condition = random.choice(CONDITION_NAMES)
    
    return selected_condition




@socketio.on('user_sent_pid')
def user_sent_pid(message):
    # resent hitid to the counselor's participant id
#     print('message: ', message)
#     locations = ['top', 'middle', 'bottom'] if message["version"] == 'distinct' else ['left', 'center', 'right']
    
    pid = message["data"]
    chosen_agent = message['agent']
    
    if chosen_agent == 'ethics_base':
        session_info[request.sid]['condition'] = chosen_agent
        selected_condition = 'ethics_base'
        first_message = '[Please write the first message]'
        
    else: 
        selected_condition = get_condition_for_participant(pid)
        first_message = CONDITION_MESSAGES[selected_condition][0]
        session_info[request.sid]['convo'].append(('START', first_message))
        session_info[request.sid]['condition'] = selected_condition + '-' + chosen_agent
    
    
    session_info[request.sid]['pid'] = pid
    condition = session_info[request.sid]['condition']
    emit('render_pid', {"data":'Participant ID "%s". Session loading first message' %  message["data"]}, room=request.sid)
        
    with sqlite3.connect('data/session_info.db') as conn:
        cur = conn.cursor()
#             sid text, pid text, condition text, message text, response text
        db_input = (request.sid, pid, condition, 'START', first_message, session_info[request.sid]['num_written'])
        cur.executemany("INSERT INTO message_pairs VALUES (?, ?, ?, ?, ?, ?)", [db_input,])
        conn.commit()
    print('INTO MESSAGE_PAIRS: ', db_input)
    
    
    emit('render_sys_message', {"data": first_message }, room=request.sid) 
    
    # self_observe the first message into the history of the agent HERE
#     first_reply = {'id': 'Gpt2', 'episode_done': False, 'text': first_message}
    if chosen_agent not in ['ethics_base', 'covid_flow']:
        session_info[request.sid]['parlai_history'].add_reply(first_message)
    
    print("participant id entered", pid)
    


def package_text(text_string): 
    return {"text": text_string, "episode_done": False}
    
    


    
def replace_last_reply(history, reply): 
    history.history_raw_strings = history.history_raw_strings[:-1]
    history.history_raw_strings.append(reply)
    
    if history.add_person_tokens:
        reply = history._add_person_tokens(reply, history.p2_token)
        
    history.history_strings = history.history_strings[:-1]
    history.history_strings.append(reply)
    
    history.history_vecs = history.history_vecs[:-1]
    history.history_vecs.append(history.parse(reply))




@socketio.on('user_sent_message_generate')   
def user_sent_message_generate(message):
    
    """
    Called when the participant sends a message to the generative model.
    """
    
    if len( message["data"].strip()) > 0:
        session_info[request.sid]['num_written'] += 1
    # message['count'] = session_info[request.sid]['num_written']
    
#     session_info[request.sid]['condition'] = 'generation'
    agent_choice = message["agent"]
    
    pid = session_info[request.sid]['pid']
    condition = session_info[request.sid]['condition']#+'-'+agent_choice
#     condition = agent_choice
    exchange_num = session_info[request.sid]['num_written']
    
    
    # Extract a string of the user's message
    raw_user_input_text = message["data"]
    flow_has_more = exchange_num < len(CONDITION_MESSAGES_NO_ACK[CONDITION_NAMES_NO_ACK[0]])
    input_not_empty = len(raw_user_input_text.strip()) > 0
    
#     message['is_done'] = 'false' if flow_has_more else 'true'
    emit('render_usr_message', message, room=request.sid)
    
        
    if flow_has_more and input_not_empty:
    
        input_text = raw_user_input_text
        
        torch.cuda.set_device(agents[message['agent']].opt['gpu'])
        
        # make sure model history is reset
        agents[agent_choice].reset()
        
        # set model history as user's conversation history
        agents[agent_choice].history = copy.deepcopy(session_info[request.sid]['parlai_history'])
        
        print('PRE-GEN: ')
        print(agents[agent_choice].history.history_strings)
        # observe user input        
        agents[agent_choice].observe(package_text(input_text))
        
        # generate bot response
        agent_output = agents[agent_choice].act()
        
        ack = normalize_reply(agent_output['text'])
        continuation = CONDITION_MESSAGES_NO_ACK[CONDITION_NAMES_NO_ACK[0]][exchange_num]
        bot_response = ack + " " + continuation
        
        
        replace_last_reply(agents[agent_choice].history, bot_response)
        
        print(agent_output)
        print('POST-GEN: ')
        print(agents[agent_choice].history.history_strings)
        
        
        # make sure to store updated user's history
        session_info[request.sid]['parlai_history'] = copy.deepcopy(agents[agent_choice].history)
        
        # make sure model history is reset
        agents[agent_choice].reset()
        
        print('POST-RESET: ')
        print(agents[agent_choice].history.history_strings)
        
#         input_text = raw_user_input_text.lower() # debug: review the preprocessing of the raw input text.
        
#         output_text, response_info = MODEL_DICT[selected_model].chat(input_text, 
#                                                                     compound_sid, 
#                                                                     '', # assignmentid
#                   
#         bot_response = CONDITION_MESSAGES[condition][exchange_num]


        output_text = bot_response                                       
   
        with sqlite3.connect('data/session_info.db') as conn:
            cur = conn.cursor()
#             sid text, pid text, condition text, message text, response text, exchange_num int
            db_input = (request.sid, pid, condition, 
                        input_text, output_text, 
                        exchange_num)
            cur.executemany("INSERT INTO message_pairs VALUES (?, ?, ?, ?, ?, ?)", [db_input])
            conn.commit()
        
        print('Into message_pairs: ', db_input)
        
#         output_text = output_text.replace('fucking', '<EXPLETIVE>').replace('fuck', '<EXPLETIVE>')
    
        # Render our response
#         time.sleep(2)
        emit('render_sys_message', {"data": output_text}, room=request.sid)
#         response_info_str = json.dumps(response_info) # debug: uncomment storing to sql below.
#         with sqlite3.connect('data/session_info.db') as conn:
#             cur = conn.cursor()
#             cur.executemany("INSERT INTO response_info VALUES (?, ?)", [(compound_sid, response_info_str,)])
#             conn.commit()
        
        session_info[request.sid]['convo'].append((raw_user_input_text, output_text))
        
#     elif not flow_has_more:
#         emit('render_sys_message', {"data": '[Chat completed. Please continue to survey.]'}, room=request.sid)
            
    elif not input_not_empty:
        emit('render_sys_message', {"data": '[oops, please enter message text]'}, room=request.sid)

    else:
        emit('render_sys_message', {"data": '[Chat completed. Please continue to survey or refresh page for a new conversation.]'}, room=request.sid)


@socketio.on('user_sent_message_interleave')   
def user_sent_message_interleave(message):
    
    """
    Called when the participant sends a message to the interleave approach.
    """
    
    if len( message["data"].strip()) > 0:
        session_info[request.sid]['num_written'] += 1
    # message['count'] = session_info[request.sid]['num_written']
    
    agent_choice = message['agent']
    flow_condition = session_info[request.sid]['condition'].split('-')[0]
#     session_info[request.sid]['condition'] = flow_condition+'-interleave'
    condition = session_info[request.sid]['condition']
    
    pid = session_info[request.sid]['pid']
    exchange_num = session_info[request.sid]['num_written']
    
    
    # Extract a string of the user's message
    raw_user_input_text = message["data"]
    flow_has_more = int(exchange_num/2) < len(CONDITION_MESSAGES[flow_condition])
    input_not_empty = len(raw_user_input_text.strip()) > 0
    
#     message['is_done'] = 'false' if flow_has_more else 'true'
    emit('render_usr_message', message, room=request.sid)
    
    if input_not_empty:  
        
        
        input_text = raw_user_input_text
        
        if not USE_BLENDER:
            torch.cuda.set_device(agents[message['agent']].opt['gpu'])
        
            # make sure model history is reset
            agents[agent_choice].reset()
        
            # set model history as user's conversation history
            agents[agent_choice].history = copy.deepcopy(session_info[request.sid]['parlai_history'])
        
            print('PRE-GEN: ')
            print(agents[agent_choice].history.history_strings)
            # observe user input        
            agents[agent_choice].observe(package_text(input_text))
        
        
        if flow_has_more and (exchange_num % 2 == 0):
            bot_response = CONDITION_MESSAGES[flow_condition][int(exchange_num/2)]
            
            if not USE_BLENDER:
                agents[agent_choice].history.add_reply(bot_response)
        
        else:        
            # generate bot response
            
            if USE_BLENDER: 
                current_user_text = input_text
                last_user_text = ''
                if len(session_info[request.sid]['convo']) > 1: # first pair is ('START', first_flow_message)
                    last_user_text = session_info[request.sid]['convo'][-1][0]
                last_bot_response = session_info[request.sid]['convo'][-1][1]
                    
                bot_response = get_blender_response(last_user_text, last_bot_response, current_user_text, key=str(time.time()))
            
            else:
                agent_output = agents[agent_choice].act()
                bot_response = normalize_reply(agent_output['text'])
                
#             print(agent_output)

#         print('POST-GEN: ')
#         print(agents[agent_choice].history.history_strings)
        
        # make sure to store updated user's history
        if not USE_BLENDER:
            session_info[request.sid]['parlai_history'] = copy.deepcopy(agents[agent_choice].history)
        
        # make sure model history is reset
        if not USE_BLENDER:
            agents[agent_choice].reset()
        
#         print('POST-RESET: ')
#         print(agents[agent_choice].history.history_strings)
        
#         input_text = raw_user_input_text.lower() # debug: review the preprocessing of the raw input text.
        
#         output_text, response_info = MODEL_DICT[selected_model].chat(input_text, 
#                                                                     compound_sid, 
#                                                                     '', # assignmentid
#                   
#         bot_response = CONDITION_MESSAGES[condition][exchange_num]


        output_text = bot_response                                       
   
        with sqlite3.connect('data/session_info.db') as conn:
            cur = conn.cursor()
#             sid text, pid text, condition text, message text, response text, exchange_num int
            db_input = (request.sid, pid, condition, 
                        input_text, output_text, 
                        exchange_num)
            cur.executemany("INSERT INTO message_pairs VALUES (?, ?, ?, ?, ?, ?)", [db_input])
            conn.commit()
        
        print('Into message_pairs: ', db_input)
        
#         output_text = output_text.replace('fucking', '<EXPLETIVE>').replace('fuck', '<EXPLETIVE>')
    
        # Render our response
#         time.sleep(2)
        emit('render_sys_message', {"data": output_text}, room=request.sid)
#         response_info_str = json.dumps(response_info) # debug: uncomment storing to sql below.
#         with sqlite3.connect('data/session_info.db') as conn:
#             cur = conn.cursor()
#             cur.executemany("INSERT INTO response_info VALUES (?, ?)", [(compound_sid, response_info_str,)])
#             conn.commit()
        
        session_info[request.sid]['convo'].append((raw_user_input_text, output_text))
        
#     elif not flow_has_more:
#         emit('render_sys_message', {"data": '[Chat completed. Please continue to survey.]'}, room=request.sid)
            
    else:
        emit('render_sys_message', {"data": '[oops, please enter message text]'}, room=request.sid)
        
        
        
        
        


@socketio.on('user_sent_message')   
def user_sent_message(message):
    """
    Called when the participant sends a message.
    """
    
    if len( message["data"].strip()) > 0:
        session_info[request.sid]['num_written'] += 1
    # message['count'] = session_info[request.sid]['num_written']
    
    pid = session_info[request.sid]['pid']
    condition = session_info[request.sid]['condition']
    flow_condition = condition.split('-')[0]
    exchange_num = session_info[request.sid]['num_written']
    
    
    # Extract a string of the user's message
    raw_user_input_text = message["data"]
    flow_has_more = exchange_num < len(CONDITION_MESSAGES[flow_condition])
    input_not_empty = len(raw_user_input_text.strip()) > 0
    
    message['is_done'] = 'false' if flow_has_more else 'true'
    emit('render_usr_message', message, room=request.sid)
    
    
    
    if flow_has_more and input_not_empty:
        
#         input_text = raw_user_input_text.lower() # debug: review the preprocessing of the raw input text.
        
#         output_text, response_info = MODEL_DICT[selected_model].chat(input_text, 
#                                                                     compound_sid, 
#                                                                     '', # assignmentid
#                   
        input_text = raw_user_input_text
        bot_response = CONDITION_MESSAGES[flow_condition][exchange_num]
        output_text = bot_response                                       
   
        with sqlite3.connect('data/session_info.db') as conn:
            cur = conn.cursor()
#             sid text, pid text, condition text, message text, response text, exchange_num int
            db_input = (request.sid, pid, condition, 
                        input_text, output_text, 
                        exchange_num)
            cur.executemany("INSERT INTO message_pairs VALUES (?, ?, ?, ?, ?, ?)", [db_input])
            conn.commit()
        
        print('Into message_pairs: ', db_input)
        
#         output_text = output_text.replace('fucking', '<EXPLETIVE>').replace('fuck', '<EXPLETIVE>')
    
        # Render our response
        time.sleep(2)
        emit('render_sys_message', {"data": output_text}, room=request.sid)
#         response_info_str = json.dumps(response_info) # debug: uncomment storing to sql below.
#         with sqlite3.connect('data/session_info.db') as conn:
#             cur = conn.cursor()
#             cur.executemany("INSERT INTO response_info VALUES (?, ?)", [(compound_sid, response_info_str,)])
#             conn.commit()
        
        session_info[request.sid]['convo'].append((raw_user_input_text, output_text))
        
    elif not flow_has_more:
        emit('render_sys_message', {"data": '[Chat completed. Please continue to survey.]'}, room=request.sid)
            
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
    
@socketio.on('log_annotations')  
def log_annotations(annotations):
    print('called')
    print(annotations)


    
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
#         pid = session_info[compound_sid]['pid'] # this is set as the participant ID in the counselor join function.
#         
#         
#         db_input = (compound_sid, joined_time, submit_time, duration, location_model_name, pid, num_written, str(SANDBOX), pid, feedback['data'])
#         cur.executemany("INSERT INTO counselor_feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [db_input,])
#         conn.commit()
# #     print('added to db: ', db_input)
#     
#     pickle_name = 'counselor_logs/feedback_counselor_%s_sid_%s_joined_%s.pkl' % (pid, compound_sid, submit_time)
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


# gunicorn --worker-class eventlet -w 1 -b 127.0.0.1:6776 'server:app(model_file="/home/oademasi/transfer-learning-conv-ai/ParlAI/trained_models/therapybot_v1_gpt2_small", skip-generation="false")'