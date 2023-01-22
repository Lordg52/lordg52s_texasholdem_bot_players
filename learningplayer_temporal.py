#variation of honestplayer that is AI enhanced 
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards
from pypokerengine.engine.hand_evaluator import HandEvaluator

import pandas as pd
import numpy as np
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
inputs = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
previous_stack = []
start_stack = 0
playernum = 0

def highest_possible_hand(community): #community is the cards in the community in the ['D2', 'DT', 'SK'] format 
	if community == []:
		return 0
	set_community = set(community)
	suits = ['S', 'H', 'C', 'D']
	values = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
	possible_hands = 0
	
	for i in suits: 
		set_royalflush = set([i+'A', i+'K', i+'Q', i+'J', i+'T'])
		j = set_royalflush.intersection(set_community)
		if len(j) >= 3: 
			possible_hands += 10000000 #royal flush
		
		set_flush = set([i+'2',i+'3',i+'4',i+'5',i+'6',i+'7',i+'8',i+'9',i+'T',i+'J',i+'Q',i+'K',i+'A'])
		j = set_flush.intersection(set_community)
		if len(j) >= 3:
			possible_hands += 10000 #flush
	
	for i in values: 
		set_fourkind = set(['S'+i,'H'+i,'C'+i,'D'+i])
		k = set_fourkind.intersection(set_community)
		if len(k) >= 3:
			possible_hands += 1000000 #four of a kind 
		if len(k) >= 2: 
			possible_hands += 100000 #full house 
	
	newcom = []
	for i in community:
		if i[1] == 'A':
			newcom.append(14)
			newcom.append(1)
		elif i[1] == 'K':
			newcom.append(13)
		elif i[1] == 'Q':
			newcom.append(12)
		elif i[1] == 'J':
			newcom.append(11)
		elif i[1] == 'T':
			newcom.append(10)
		else: 
			newcom.append(int(i[1]))
	newcom.sort(reverse=True)
	for i in range(2):
		try:
			newcom[i+2]
		except: 
			break

		j = newcom[i] - newcom[i+2] 
		if j <= 5:
			possible_hands += 1000 #straight
	
	return possible_hands


def loss_function(y_true, y_pred): 
	'''
	#y_true is the amount won, y_pred is the amount bet
	#since y_pred is the amount bet, we cannot use leaky ReLU as negative amounts will skew the data improperly. 
	#however: if we use the prediction of 0 to represent a fold, our loss will almost always return 0, indicating no further reduction can be made. 
	#thus we will utilize leaky relu and return the absolute value of the subtraction: 
	#a strong fold will be a very large negative, returning a positive value to minimize 
	#a call will return 0, returning a 0 if the round was lost and the negative win if the round was won 
	#a raise will be a large positive, returning a positive value to minimize 
	'''
	prediction_sum = tf.reduce_sum(y_pred[:, -5:])
	return (tf.abs(prediction_sum) - y_true) #the net will attempt to minimize negative wins, or in other words, maximize the amount won

if os.path.exists(r'saved_model/learningplayer_temporal'): 
	model = tf.keras.models.load_model('saved_model/learningplayer_temporal', custom_objects={'loss_function': loss_function})
else:
	model = keras.Sequential([
			layers.Conv1D(10, 5, input_shape=(5, 10), activation="tanh", name="layer_in"),
			layers.Dense(32, activation="LeakyReLU", name="layer1"),
			layers.Dense(8, activation="LeakyReLU", name="layer2"),
			layers.Dense(1, activation="tanh", name="layer_out")])
	opt = keras.optimizers.Adam(learning_rate=0.01)
	model.compile(loss=loss_function, optimizer=opt)
model.summary()


def determine_handstrength(hole_cards, community_cards):
	hand_info = HandEvaluator.gen_hand_rank_info(hole_cards, community_cards)
	strength = HandEvaluator.eval_hand(hole_cards, community_cards)
	return hand_info, strength


class learningplayer_temporal(BasePokerPlayer):

	def declare_action(self, valid_actions, hole_card, round_state):
		global inputs, playernum
		community_card = round_state['community_card'] 
		
		print('display round state:', round_state)
		dataframe_round_seats = pd.DataFrame(round_state['seats'])
		pokerbot_input1 = dataframe_round_seats['stack'].to_numpy()
		pokerbot_input2 = [determine_handstrength(hole_cards=gen_cards(hole_card), community_cards=gen_cards(community_card))[-1], highest_possible_hand(community_card), round_state['dealer_btn'], round_state['next_player']]
		self.pokerbot_inputs = np.append(pokerbot_input1, pokerbot_input2)
		inputs.append(self.pokerbot_inputs.tolist())
		
		print(self.pokerbot_inputs)
		inputs.pop(0)
		print(np.asarray(inputs).shape)
		print(inputs)
		
		prediction = model.predict(np.asarray(inputs).reshape(1, 5, 10))
		print(prediction)
		print(valid_actions)

		if prediction[0][0][0]*valid_actions[2]['amount']['max'] > valid_actions[2]['amount']['min']:
			action = valid_actions[2]  # fetch RAISE action info
			return action['action'], prediction[0][0][0]*valid_actions[2]['amount']['max']
		elif prediction[0][0][0] < 0: #fold
			action = valid_actions[0]
			foldstate = True
		else:
			action = valid_actions[1]  # fetch CALL action info
			prediction = valid_actions[1]['amount']
		
		playernum = round_state['next_player']
		return action['action'], action['amount']



	def receive_game_start_message(self, game_info):
		global start_stack
		self.nb_player = game_info['player_num']
		print('game info:', game_info['rule']['initial_stack'])
		start_stack = game_info['rule']['initial_stack']

	def receive_round_start_message(self, round_count, hole_card, seats):
		global previous_stack
		dataframe_startofround_seats = pd.DataFrame(seats)
		previous_stack = dataframe_startofround_seats['stack'].to_numpy()
		print('previous stack updated!', previous_stack)
		pass

	def receive_street_start_message(self, street, round_state):
		pass

	def receive_game_update_message(self, action, round_state):
		pass

	def receive_round_result_message(self, winners, hand_info, round_state):
		global inputs, previous_stack, start_stack, playernum
		
		dataframe_endofround_seats = pd.DataFrame(round_state['seats'])
		endofround_stack = dataframe_endofround_seats['stack'].to_numpy()
		round_difference_stack = endofround_stack - previous_stack
		print(endofround_stack, previous_stack, round_difference_stack)
		
		amt_won = round_difference_stack[0]
		previous_stack = endofround_stack
		
		model.fit(x=np.asarray(inputs).reshape(1, 5, 10), y=np.array([float(amt_won/start_stack)]))
		
		model.save(f'saved_model/learningplayer_temporal{playernum}')
		inputs = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
		pass

def setup_ai():
	return learningplayer_temporal()
