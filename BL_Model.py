from IPython.display import display
import numpy as np
import pandas as pd
import openpyxl
from numpy.linalg import inv
class Equilibrium_Model():
	def __init__(self, risk_aversion, excess_return_covariance, market_capitalization_weight):
		self.risk_aversion = risk_aversion
		self.excess_return_covariance = excess_return_covariance
		self.market_capitalization_weight = market_capitalization_weight
		self.equilibrium_excess_return = None
		self.update_model()



	def update_model(self):
		if (self.risk_aversion is not None and self.excess_return_covariance is not None and self.market_capitalization_weight is not None):
			self.equilibrium_excess_return = self.risk_aversion * self.excess_return_covariance @ self.market_capitalization_weight
		else:
			raise ValueError("parameter not fully specified")

class Views():
	def __init__(self, te, asset_covariance,measure_view_variance = "proportional"):
		measure_methods = ["Idzorek", "proportional"]
		if (measure_view_variance not in measure_methods):
			raise ValueError("measure method should be in " + str(measure_methods))
		self.measure_view_variance = measure_view_variance
		self.p = None
		self.q = None
		self.view_covariance = None
		self.te = te
		self.asset_covariance = asset_covariance
		self.view_descriptions = []
		self.confidence = []


	def calculate_view_variance(self):
		if (self.measure_view_variance == "Idzorek"):
			m = self.p @ (self.te * self.asset_covariance) @ self.p.transpose()
			for i in range(len(self.confidence)):
				alpha = (1 - self.confidence[i]) / self.confidence[i]
				m[i][i] = alpha * m[i][i]
			self.view_covariance = np.diag(np.diag(m))
		elif (self.measure_view_variance == "proportional"):
			m = self.p @ (self.te * self.asset_covariance) @ self.p.transpose()
			self.view_covariance = np.diag(np.diag(m))


class BL_model():
	def __init__(self, assets_name, te, market_capitalization_weight, risk_aversion, excess_return_covariance, canonical=True, measure_view_variance="proportional", eqully_weighted=True):
		self.assets_name = assets_name	
		#self.assets_data = None
		self.equilibrium_model = Equilibrium_Model(risk_aversion, excess_return_covariance, market_capitalization_weight)
		self.views = Views(te, excess_return_covariance, measure_view_variance)
		self.te = te
		self.canonical = canonical
		self.prior_weight = self.calculate_prior_weight(market_capitalization_weight)
		self.posterior_return = None
		self.posterior_weight = None
		self.posterior_variance = None
		self.eqully_weighted = eqully_weighted

	def calculate_prior_weight(self, market_capitalization_weight):
		if (self.canonical):
			return market_capitalization_weight / (1 + self.te)
		else:
			return market_capitalization_weight

	def add_abosolute_view(self, asset, num,  content="", confidence=None):
		if isinstance(asset, str):
			index = self.assets_name.index(asset)

		elif isinstance(asset, int):
			index = asset
		else:
			raise ValueError("specified asset not should be integer or string")
		if (self.views.measure_view_variance =="Idzorek" and  confidence is None):
			raise ValueError("confidence needs to be specified in Idzorek measure method")
		self.views.confidence.append(confidence)
		row = np.zeros(len(self.assets_name))
		row[index] = 1
		if (self.views.p is None and self.views.q is None):
			self.views.p = row.reshape(1, len(self.assets_name))
			self.views.q = np.array([num]).reshape(1,1)
		else:
			self.views.p = np.vstack((self.views.p, row))
			self.views.q = np.vstack((self.views.q, num))
		self.views.view_descriptions.append(content)
		self.views.calculate_view_variance()
	def add_relative_view(self, assets1, assets2, num, content="", confidence=None):
		#use equal weights
		if (self.views.measure_view_variance =="Idzorek" and  confidence is None):
			raise ValueError("confidence needs to be specified in Idzorek measure method")
		self.views.confidence.append(confidence)
		row = np.zeros(len(self.assets_name))

		total_weight1 = 0
		for index in assets1:
			if isinstance(index, str):
				index = self.assets_name.index(index)
			total_weight1 += self.equilibrium_model.market_capitalization_weight[index]

		for index in assets1:
			if isinstance(index, str):
				index = self.assets_name.index(index)
			if self.eqully_weighted:
				weight1 = 1 / len(assets1)
			else:
				weight1 = self.equilibrium_model.market_capitalization_weight[index] / total_weight1	
			row[index] = weight1

		total_weight2 = 0
		for index in assets2:
			if isinstance(index, str):
				index = self.assets_name.index(index)
			total_weight2 += self.equilibrium_model.market_capitalization_weight[index]	

		for index in assets2:
			if isinstance(index, str):
				index = self.assets_name.index(index)
			if (self.eqully_weighted):
				weight2 = (-1) / len(assets2)
			else:
				weight2 =  -self.equilibrium_model.market_capitalization_weight[index] / total_weight2
			row[index] = weight2

		if (self.views.p is None and self.views.q is None):
			self.views.p = row.reshape(1, len(self.assets_name))
			self.views.q = np.array([num]).reshape(1,1)
		else:			
			self.views.p = np.vstack((self.views.p, row))
			self.views.q = np.vstack((self.views.q, num))
		self.views.view_descriptions.append(content)
		self.views.calculate_view_variance()

	def calculate_posterior_return(self):
		first_term = inv(self.te * self.equilibrium_model.excess_return_covariance)
		second_term = self.views.p.transpose() @ inv(self.views.view_covariance) @ self.views.p
		third_term = inv(self.te * self.equilibrium_model.excess_return_covariance) @ self.equilibrium_model.equilibrium_excess_return
		fourth_term = self.views.p.transpose() @ inv(self.views.view_covariance) @ self.views.q
		self.posterior_return =  inv(first_term + second_term) @ (third_term + fourth_term) 

	def calculate_posterior_variance(self):
		if (self.canonical):
			first_term = inv(self.te * self.equilibrium_model.excess_return_covariance)
			second_term = self.views.p.transpose() @ inv(self.views.view_covariance) @ self.views.p
			self.posterior_variance =  inv(first_term + second_term) + self.equilibrium_model.excess_return_covariance
		else:
			self.posterior_variance = self.equilibrium_model.excess_return_covariance

	def calculate_posterior_weight(self):
		self.posterior_weight =  inv(self.equilibrium_model.risk_aversion * self.posterior_variance) @ self.posterior_return

	# def clear_view(self):
	# 	self.views.p = None
	# 	self.views.q = None
	# 	self.views.view_covariance = None
	# 	self.descriptions = []

	def build_model(self):
		self.calculate_posterior_return()
		self.calculate_posterior_variance()
		self.calculate_posterior_weight()

	def print_summary(self):
		self.build_model()
		data = {"CAPM平衡超额收益率":self.equilibrium_model.equilibrium_excess_return.flatten(), 
		"CAPM平衡投资权重":self.calculate_prior_weight(self.equilibrium_model.market_capitalization_weight).flatten(),
		"BL模型超额收益率": self.posterior_return.flatten(),
		"BL模型投资权重": self.posterior_weight.flatten()}
		df = pd.DataFrame(data = data, index = self.assets_name)
		if (self.canonical):
			print("模型：经典Black-litterman参考模型")
		else:
			print("模型：另类Black-litterman参考模型")
		if (self.views.measure_view_variance == "proportional"):
			print("测量观点误差的方法: 与资产方差成比例")
		elif(self.views.measure_view_variance == "Idzorek"):
			print("测量观点误差的方法：Idzorek的观点信心法")
		if (self.eqully_weighted):
			print("观点权重：均分")
		else:
			print("观点权重：根据市场份额")
		display(df)
		print("（单位：百分比 %）")
		if (self.views.measure_view_variance == "Idzorek"):
			views_data = {"描述":self.views.view_descriptions, "信心": self.views.confidence}
		else:
			views_data = {"描述":self.views.view_descriptions}
		view_df = pd.DataFrame(data=views_data, index=['观点' + str(i) for i in range(1, len(self.views.view_descriptions) + 1)])
		display(view_df)

	def print_to_excel(self):
		self.build_model()
		data = {"CAPM平衡超额收益率":self.equilibrium_model.equilibrium_excess_return.flatten(), 
		"CAPM平衡投资权重":self.calculate_prior_weight(self.equilibrium_model.market_capitalization_weight).flatten(),
		"BL模型超额收益率": self.posterior_return.flatten(),
		"BL模型投资权重": self.posterior_weight.flatten()}
		df = pd.DataFrame(data = data, index = self.assets_name)
		df.round(2).to_excel("BL模型数据.xlsx", sheet_name="总结")
		if (self.views.measure_view_variance == "Idzorek"):
			views_data = {"描述":self.views.view_descriptions, "信心": self.views.confidence}
		else:
			views_data = {"描述":self.views.view_descriptions}
		view_df = pd.DataFrame(data=views_data, index=['观点' + str(i) for i in range(1, len(self.views.view_descriptions) + 1)])		
		view_df.to_excel("BL模型数据.xlsx", sheet_name="观点")


