from agent.ai_trader import AiTrader
from helper_functions import *
from tqdm import tqdm_notebook, tqdm


window_size = 10
episodes = 1000
batch_size = 32

# The data
stock_name = "AAPL"
data = dataset_loader(stock_name)
data_samples = len(data) - 1

# The AI trader
trader = AiTrader(window_size)
trader.model.summary()

for episode in range(1, episodes + 1):

	print("Episode: {}/{}".format(episode, episodes))
	state = state_creator(data, 0, window_size + 1)
	total_profit = 0
	trader.inventory = []

	for t in tqdm(range(data_samples)):

		action = trader.trade(state)
		next_state = state_creator(data, t + 1, window_size + 1)
		reward = 0

		if action == 1:  # Buying
			trader.inventory.append(data[t])
			print("AI Trader bought: ", stocks_price_format(data[t]))

		elif action == 2 and len(trader.inventory) > 0:  # Selling
			buy_price = trader.inventory.pop(0)
			sold_price = data[t]
			current_profit = sold_price - buy_price
			reward = max(current_profit, 0)
			# Calculate total profit
			total_profit += current_profit
			print("AI Trader sold: ", stocks_price_format(data[t]), " Profit: " + stocks_price_format(current_profit))

		if t == data_samples - 1:
			done = True
		else:
			done = False

		trader.memory.append((state, action, reward, next_state, done))
		state = next_state
		if len(trader.memory) > batch_size:
			trader.batch_train(batch_size)

		if done:
			print("########################")
			print("TOTAL PROFIT: {}".format(total_profit))
			print("########################")

	if episode % 10 == 0:
		trader.model.save("ai_trader_{}.h5".format(episode))
