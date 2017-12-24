from operator import itemgetter
import string
import copy
### Debug #############3
from PIL import Image, ImageDraw, ImageFont




class SymbolManager:
	def __init__(self):
		
		print('initializing symbols ...')

		# structure of an entry:
		# (sym, index, class, Alignment)
		#
		#
		self.dictionary = []
		
		
		try:
			with open('label.txt') as f:
				sym_list = f.readlines()
				for sym in sym_list:
					temp = sym.replace('\n', '').split(' ')
					self.dictionary.append([temp[0], int(temp[1])])

			
					
		except FileNotFoundError as e:
			print('Error label.txt is not found')
			return

		self.AddClassToDictionary()
		#print(self.dictionary)

	def AddClassToDictionary(self):
		# Class consist of 
		#  NonScripted: + - = > ->
		NC = ['add', 'sub', '=', 'rightarrow', 'leq', 'geq', 'neq'] #7
		#  Bracket ( { [
		BR = ['(']
		#  Root : sqrt
		SQ = ['sqrt']
		#  VariableRange : _Sigma, integral, _PI, 
		VR = ['_Sigma', '_Pi', 'lim', 'integral']
		#  Plain_Ascender: 0..9, A..Z, b d f h i k l t
		PA = ['b', 'd', 'f', 'h', 'i', 'k', 'l', 't', 'exists', 'forall', '!', '_Delta', '_Omega', '_Phi', 'div', 'beta', 'lamda', 'tan', 'log']
		PA = PA + list(map(str, range(10)))
		PA = PA + list(string.ascii_uppercase)
		#  Plain_Descender: g p q y gamma, nuy, rho khi phi
		PD = ['g', 'p', 'q', 'y', 'gamma', 'muy', 'rho', '']
		#  plain_Centered: The rest
		for entry in self.dictionary:
			temp_class = 'plain_Centered'
			Alignment = 'Centred'
			if entry[0] in NC:
				temp_class = 'NonScripted'
			elif entry[0] in BR:
				temp_class = 'Bracket'
			elif entry[0] in SQ:
				temp_class = 'Root'
			elif entry[0] in VR:
				temp_class = 'VariableRange'
			elif entry[0] in PA:
				temp_class = 'Plain_Ascender'
				Alignment = 'Ascender'
			elif entry[0] in PD:
				temp_class = 'Plain_Descender'
				Alignment = 'Descender'
			entry.append(temp_class)
			entry.append(Alignment)
			
	def getClass(self, symbol):
		for entry in self.dictionary:
			if entry[0] == symbol:
				return entry[2]
		
	def getSymbolFromIndex(self, index):
		for entry in self.dictionary:
			if entry[1] == index:
				return entry[0], entry[2], entry[3]


				
class LBST:
	def __init__(self, psymbol_manager):
		root = []
		self.symbol_manager = psymbol_manager

		self.compoundable_list = list('0123456789abcdefghijklmnopqrstuvwz') + ['ldot']
		self.Prefix_compoundable_list = list('ABCDEFGHIJKLMNOPQRSTUVWZ')
		self.OperatorList_eq = ['=', 'neq', 'in', 'geq', 'leq']
		self.OperatorList_as = ['add', 'sub']
		self.OperatorList_md = ['time', 'div', 'slash']

		self.AllOp = self.OperatorList_eq + self.OperatorList_as + self.OperatorList_md 

		self.AllowAjacentAsMultiply = ['sub', 'sup', 'literal', 'sqrt'] #please handle f(x)

		self.AllowAjacentAsMultiplyRight = ['sin', 'cos', 'tan', 'log', 'lim']

		self.VariableRangeList = ['_Sigma', '_Pi', 'integral', 'lim', 'frac', 'rightarrow']

		# RULE TO PARSE :
		#:
		#SUM
		#PI
		#     sub sup 
		#frac
		#int
		#     sqrt
		#lim
		#rightarrow


		#child_temp: TL, BL, T, B, C, SUP, SUB

		############################
		# LBSTnode_sym
		# sym, BST_node

	def process(self, BSTtree):
	

	
		try:
			LBSTtree = self.createLBSTtreefromBSTtree(BSTtree)
			return LBSTtree
		except:
			print('unable to parse')
			print(BSTtree)
			return []

		#OperatorTree = self.createOperatorTreeFromLBSTTree(LBSTtree)
		#print(OperatorTree)
		#return

		try:
			OperatorTree = self.createOperatorTreeFromLBSTTree(LBSTtree)
			print(OperatorTree)
		except:
			print('unable to parse')
			print(LBSTtree)

		

	def createLiteralNode(self, sym):
		node = {}

		if len(sym) == 2 and sym[0] == 'z':
			sym = sym[1]	

		node['symbol'] = sym
		node['type'] = 'literal'

		if sym in self.OperatorList_as or sym in self.OperatorList_md or sym in self.OperatorList_eq:
			node['type'] = 'Operation'

		return node

	def createParentNode(self, ntype):
		node = {}
		node['type'] = ntype

		return node

	def deleteOldChild(self, tree):
		for node in tree:

			if 'child' in node:
				for child in node['child']:
					self.deleteOldChild(child)

			if 'old_child' in node:
				del node['old_child'] 

	def createCompoundSymbolBaseline(self, BSTtree):
		compounded_tree = []
		merging = False

		numlist = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'ldot']

		for node in BSTtree:
			sym, clas, align = self.symbol_manager.getSymbolFromIndex(node[4])


			if merging == True and sym in self.compoundable_list or (sym == 'ldot' and compounded_tree[-1]['symbol'][-1] in numlist):

				if compounded_tree[-1]['symbol'][-1] in numlist and sym not in numlist:
					compounded_tree.append(self.createLiteralNode(sym))
				else:
					compounded_tree[-1]['symbol'] = compounded_tree[-1]['symbol'] + sym
				
				if self.countChild(node) != 0:
					merging = False

					compounded_tree[-1]['old_child'] = node[9]

					#if len(node[9][6]) > 0: #sub
					#	temp_node = self.createParentNode('child_sub')	
					#	temp_node['base'] = compounded_tree[-1]['symbol']				
					#	temp_node['sub_sym'] = self.createCompoundSymbolBaseline(node[9][6])
					#compounded_tree.pop()
					#compounded_tree.append(temp_node)
			else:
				merging = False

				temp_node = self.createLiteralNode(sym)

				temp_node['old_child'] = node[9]
				compounded_tree.append(temp_node)
				if self.countChild(node) == 0:
					if sym in self.compoundable_list or sym in self.Prefix_compoundable_list:
						merging = True

		return compounded_tree

	def handleSub(self, BSTtree):
		for idx in range(len(BSTtree)):

			if 'old_child' not in BSTtree[idx]:
				continue

			#SUB
			if len(BSTtree[idx]['old_child'][6]) > 0:

				newnode = self.createParentNode('sub')
				newnode['child'] = []
				newnode['child'].append([BSTtree[idx]])

				#print('zzzzzzzzzzzzzzzzzzz')
				#if (BSTtree[idx]['symbol'] == 'B'):
				#	print(BSTtree[idx]['old_child'])

				newnode['child'].append(self.createLBSTtreefromBSTtree(BSTtree[idx]['old_child'][6]))

				BSTtree[idx]['old_child'][6] = []
				newnode['old_child'] = BSTtree[idx]['old_child']

				del BSTtree[idx]
				BSTtree.insert(idx, newnode)


	def handleSup(self, BSTtree):

		#print(BSTtree)

		for idx in range(len(BSTtree)):

			if 'old_child' not in BSTtree[idx]:
				continue

			#SUB
			#if len(BSTtree[idx]['old_child'][6]) > 0:
			#	BSTtree[idx]['type'] = 'sub'
			#	BSTtree[idx]['child'] = []
			#	BSTtree[idx]['child'].append([self.createLiteralNode(BSTtree[idx]['symbol'])])
			#	del BSTtree[idx]['symbol']
			#	BSTtree[idx]['child'].append(self.createLBSTtreefromBSTtree(BSTtree[idx]['old_child'][6]))
			#	BSTtree[idx]['old_child'][6] = []

			#SUB
			#SUP
			if len(BSTtree[idx]['old_child'][5]) > 0:

				newnode = self.createParentNode('sup')
				newnode['child'] = []
				newnode['child'].append([BSTtree[idx]])
				newnode['child'].append(self.createLBSTtreefromBSTtree(BSTtree[idx]['old_child'][5]))

				BSTtree[idx]['old_child'][5] = []
				newnode['old_child'] = BSTtree[idx]['old_child']

				del BSTtree[idx]
				BSTtree.insert(idx, newnode)

	def handleSQRT(self, BSTtree):
		for idx in range(len(BSTtree)):

			if 'old_child' not in BSTtree[idx]:
				continue

			if len(BSTtree[idx]['old_child'][4]) > 0 and BSTtree[idx]['symbol'] == 'sqrt':
				newnode = self.createParentNode('sqrt')
				newnode['child'] = []
				newnode['child'].append(self.createLBSTtreefromBSTtree(BSTtree[idx]['old_child'][4]))

				BSTtree[idx]['old_child'][4] = []
				newnode['old_child'] = BSTtree[idx]['old_child']

				del BSTtree[idx]
				BSTtree.insert(idx, newnode)

	def handleNonScript_and_VariableRange(self, BSTtree):

		for idx in range(len(BSTtree)):

			if 'old_child' not in BSTtree[idx] or 'symbol' not in BSTtree[idx]:
				continue

				#SUM
				#PI
				#frac
				#int
				#lim
				#rightarrow

			if len(BSTtree[idx]['old_child'][3]) > 0 or len(BSTtree[idx]['old_child'][2]) > 0:
				if BSTtree[idx]['symbol'] == '_Sigma':
					newnode = self.createParentNode('_Sigma')
				elif BSTtree[idx]['symbol'] == '_Pi':
					newnode = self.createParentNode('_Pi')
				elif BSTtree[idx]['symbol'] == 'integral ':
					newnode = self.createParentNode('integral')
				elif BSTtree[idx]['symbol'] == 'lim':
					newnode = self.createParentNode('lim')
				elif BSTtree[idx]['symbol'] == 'sub':
					newnode = self.createParentNode('frac')
				elif BSTtree[idx]['symbol'] == 'rightarrow':
					newnode = self.createParentNode('rightarrow')
				else:
					newnode = self.createParentNode(BSTtree[idx]['symbol'])
				###############3


				newnode['child'] = []
				newnode['child'].append(self.createLBSTtreefromBSTtree(BSTtree[idx]['old_child'][2]))
				newnode['child'].append(self.createLBSTtreefromBSTtree(BSTtree[idx]['old_child'][3]))

				BSTtree[idx]['old_child'][2] = []
				BSTtree[idx]['old_child'][3] = []
				newnode['old_child'] = BSTtree[idx]['old_child']

				del BSTtree[idx]
				BSTtree.insert(idx, newnode)

	def handleBracket(self, BSTtree):

		open_index = -1

		for idx in range(len(BSTtree)):

			if 'symbol' not in BSTtree[idx]:
				continue



			if BSTtree[idx]['symbol'] == '(':
				open_index = idx
			elif BSTtree[idx]['symbol'] == ')':
				close_index = idx

				del_list = list(range(open_index, close_index + 1))
				del_list.reverse()

				bracket_content = [BSTtree[open_index + 1: close_index]]

				newnode = self.createParentNode('bracket')
				newnode['child'] = [self.handleCompounded_tree(bracket_content[0])]

				if 'old_child' in BSTtree[idx]:
					newnode['old_child'] = BSTtree[idx]['old_child']

				for i in del_list:
					del BSTtree[i]

				BSTtree.insert(open_index, newnode)

				return True
		return False


	def createLBSTtreefromBSTtree(self, BSTtree):

		compounded_tree = self.createCompoundSymbolBaseline(BSTtree)
		
		return self.handleCompounded_tree(compounded_tree)

	def handleCompounded_tree(self, compounded_tree):


		while self.handleBracket(compounded_tree):
			pass



		self.handleSub(compounded_tree)
		self.handleSup(compounded_tree)

		self.handleSQRT(compounded_tree)



		self.handleNonScript_and_VariableRange(compounded_tree)



		self.deleteOldChild(compounded_tree)

		return compounded_tree


	def createOperationParentNode(self, ntype):
		node = {}
		node['type'] = 'Op' + ntype

		return node

	def createFunctionParentNode(self, ntype):
		node = {}
		node['type'] = ntype

		return node

	def ParseToOperationTreeBinary(self, LBSTTree, op_list):

		idx_list = list(range(len(LBSTTree)))
		idx_list.reverse()

		for idx in idx_list:
			if 'symbol' not in LBSTTree[idx]:
				continue

			if LBSTTree[idx]['symbol'] in op_list:
				newnode = self.createOperationParentNode(LBSTTree[idx]['symbol'])
				newnode['child'] = []
				
				newnode['child'].append([LBSTTree[idx - 1]])
				newnode['child'].append([LBSTTree[idx + 1]])

				del LBSTTree[idx + 1]
				del LBSTTree[idx]
				del LBSTTree[idx - 1]

				LBSTTree.insert(idx - 1, newnode)
				return True

		return False


	def ParseToOperationTreeAS(self, LBSTTree):
		idx_list = list(range(len(LBSTTree)))
		idx_list.reverse()

		for idx in idx_list:
			if 'symbol' not in LBSTTree[idx]:
				continue

			if LBSTTree[idx]['symbol'] in self.OperatorList_as:

				newnode = self.createOperationParentNode(LBSTTree[idx]['symbol'])


				if idx == 0 or ('symbol' in LBSTTree[idx - 1] and (LBSTTree[idx - 1]['symbol'] in self.OperatorList_as or LBSTTree[idx - 1]['symbol'] in self.OperatorList_eq)): #unary
					newnode['child'] = []

					newnode['child'].append([LBSTTree[idx + 1]])

					
					del LBSTTree[idx + 1]
					del LBSTTree[idx]
					
					
					LBSTTree.insert(idx, newnode)
					return True

				else: #binary
					
					newnode['child'] = []
					newnode['child'].append([LBSTTree[idx - 1]])
					newnode['child'].append([LBSTTree[idx + 1]])


					del LBSTTree[idx + 1]
					del LBSTTree[idx]
					del LBSTTree[idx - 1]
					LBSTTree.insert(idx - 1, newnode)
					return True

		return False

	def AddInvisibleMultiply(self, LBSTTree): # for cases such as '2a' 
		idx_list = list(range(len(LBSTTree) - 1))


		for idx in idx_list:

			cond_left = LBSTTree[idx]['type'] in self.AllowAjacentAsMultiply and LBSTTree[idx]['symbol'] not in self.AllowAjacentAsMultiplyRight



			cond_right = LBSTTree[idx + 1]['type'] in self.AllowAjacentAsMultiply or ('symbol' in LBSTTree[idx + 1] and LBSTTree[idx + 1]['symbol'] in self.AllowAjacentAsMultiplyRight)

			if cond_left:

				if cond_right:

				
					newnode = self.createOperationParentNode('Otime')
					newnode['child'] = []
					newnode['child'].append([LBSTTree[idx]])
					newnode['child'].append([LBSTTree[idx + 1]])

					del LBSTTree[idx + 1]
					del LBSTTree[idx]

					LBSTTree.insert(idx, newnode)
					return True

		return False

	def ParseChild(self, LBSTTree):
		idx_list = list(range(len(LBSTTree)))

		for idx in idx_list:

			if 'child' not in LBSTTree[idx]:
				continue

			for i in range(len(LBSTTree[idx]['child'])):
				LBSTTree[idx]['child'][i] = self.ParseToOperationTree(LBSTTree[idx]['child'][i])

	def parseFunctionOperator(self, LBSTTree): #for cases such as f(x)
		idx_list = list(range(len(LBSTTree) - 1))
		idx_list.reverse()

		for idx in idx_list:

			cond_function_name = 'symbol' in LBSTTree[idx] and LBSTTree[idx]['symbol'] in self.AllowAjacentAsMultiplyRight
			cond_user_defined_name = 'symbol' in LBSTTree[idx] and LBSTTree[idx]['symbol'] not in self.AllOp #not merge to cond_function_name because i am not so sure about this case
			
			cond_left = cond_user_defined_name or cond_function_name or LBSTTree[idx]['type'] in self.VariableRangeList

			type_right = LBSTTree[idx + 1]['type']
			if type_right in self.AllOp or (len(type_right) > 1 and type_right[:2] == 'Op'):
				continue
			

			if cond_left:
				if cond_function_name or cond_user_defined_name:
					newnode = self.createFunctionParentNode('f' + LBSTTree[idx]['symbol'])
				else:
					newnode = self.createOperationParentNode('f' + LBSTTree[idx]['type'])
				newnode['child'] = []

				if 'child' in LBSTTree[idx]:
					for c in LBSTTree[idx]['child']:
						newnode['child'].append(c)

				newnode['child'].append([LBSTTree[idx + 1]])

				del LBSTTree[idx + 1]
				del LBSTTree[idx]

				LBSTTree.insert(idx, newnode)
				return True

		return False


	def ParseToOperationTree(self, LBSTTree):

		self.ParseChild(LBSTTree)
		
		while self.AddInvisibleMultiply(LBSTTree):
			pass
		
		while self.parseFunctionOperator(LBSTTree):
			pass

		while self.ParseToOperationTreeBinary(LBSTTree, self.OperatorList_md):
			pass
		while self.ParseToOperationTreeAS(LBSTTree):
			pass
		while self.ParseToOperationTreeBinary(LBSTTree, self.OperatorList_eq):
			pass

		return LBSTTree

	def createOperatorTreeFromLBSTTree(self, LBSTTree):

		return self.ParseToOperationTree(LBSTTree)

	def createLBSTnodefromBSTnode(self, BSTnode):
		sym, clas, align = self.symbol_manager.getSymbolFromIndex(BSTnode[4])


	def countChild(self, BSTnode):
		childnodes = BSTnode[9]
		count = 0
		for child in childnodes:
			count += len(child)

		return count

class LatexGenerator:
	def __init__(self):
		self.greek_alphabet = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'muy', 'rho', 'nuy', 'omega', 'lamda', 'phi', 'sigma', 'theta', 'pi']

	def process(self, LBSTTree):
		print(LBSTTree)
		output = self.createLatexString(LBSTTree)
		
		print(output)
		
		return output
		
	def getStringFromNode(self, node):
		if 'type' in node:
			if node['type'] == 'literal':
			
				if node['symbol'] == 'integral':
					return '\\int'
				if node['symbol'] == '_Delta':
					return '\\Delta'
			
				if node['symbol'] == 'ldots':
					return '\\ldots'
				if node['symbol'] == 'rightarrow':
					return ' \\rightarrow '
				if node['symbol'] == 'intft':
					return ' \\inf '
			
				node['symbol'] = node['symbol'].replace('ldot', '.')
				
				if node['symbol'] in self.greek_alphabet:
					if node['symbol'] == 'muy':
						node['symbol'] = '\\mu'
					else:
						node['symbol'] = '\\' + node['symbol']
				
				return ' ' + node['symbol'] + ' '
			elif node['type'] == 'Operation':
				if node['symbol'] == 'add':
					return ' + '
				if node['symbol'] == 'time':
					return '\\times '
				if node['symbol'] == 'sub':
					return ' - '
				if node['symbol'] == 'neq':
					return ' \\neq '
				if node['symbol'] == 'geq':
					return ' \\geq '
				if node['symbol'] == 'leq':
					return ' \\leq '
				if node['symbol'] == 'in':
					return ' \\in '
				if node['symbol'] == 'div':
					return ' \\div '
				if node['symbol'] == 'rightarrow':
					return ' \\rightarrow '


				return node['symbol']
			elif node['type'] == 'bracket':
				return '(' + self.createLatexString(node['child'][0]) + ')'
			elif node['type'] == 'sqrt':
				return '\sqrt{' + self.createLatexString(node['child'][0]) + '}'
			elif node['type'] == 'sup':
				
				
				
				base = self.createLatexString(node['child'][0])
				sup = self.createLatexString(node['child'][1])
				
				####### SPECIAL CASES ##################
				

				
				########################################
				
				return base + '^{ ' + sup + ' }'
			
			elif node['type'] == 'sub':
				
				base = self.createLatexString(node['child'][0])
				sub = self.createLatexString(node['child'][1])
				
				########### SPECIAL #############
				
				if 'symbol' in node['child'][1][0] and node['child'][1][0]['symbol'] == '.':
					return base + sub 
					
				##################################
				
				return base + '_{ ' + sub + ' }'

			elif node['type'] == 'frac':
				upper_node = self.createLatexString(node['child'][0])
				lower_node = self.createLatexString(node['child'][1])
			
				return '\\frac{' + upper_node + '}{' + lower_node + '}'
			
			elif node['type'] == '_Sigma':
				upper_node = self.createLatexString(node['child'][0])
				lower_node = self.createLatexString(node['child'][1])
			
				return '\\sum^{' + upper_node + '}_{' + lower_node + '}'
			
			elif node['type'] == '_Pi':
				upper_node = self.createLatexString(node['child'][0])
				lower_node = self.createLatexString(node['child'][1])
			
				return '\\prod^{' + upper_node + '}_{' + lower_node + '}'
		
			elif node['type'] == 'lim':
				lower_node = self.createLatexString(node['child'][1])
			
				return '\\lim_{' + lower_node + '}'
		
			elif node['type'] == 'integral':
				upper_node = self.createLatexString(node['child'][0])
				lower_node = self.createLatexString(node['child'][1])
			
				return '\\int_{' + upper_node + '}^{' + lower_node + '}'
			

			
			
			else:
				if len(node['child'][0]) > 0 and len(node['child'][1]) == 0:
					return node['type'] + '^{' + self.createLatexString(node['child'][0]) + '}'
					
				elif len(node['child'][0]) == 0 and len(node['child'][1]) > 0:
					return node['type'] + '_{' + self.createLatexString(node['child'][1]) + '}'
					
				elif len(node['child'][0]) > 0 and len(node['child'][1]) > 0:
					return node['type'] + '^{' + self.createLatexString(node['child'][0]) + '}_{' + self.createLatexString(node['child'][1]) + '}'
					
				return str(node)
		
			
		else:
			return str(node)
		
	def createLatexString(self, LBSTTree):
		Latex_string = ''
		
		if type(LBSTTree) == dict:
		
			Latex_string = Latex_string + self.getStringFromNode(LBSTTree)
		
			if 'child' in LBSTTree:
				pass
			return Latex_string
		
		for child in LBSTTree:		
			Latex_string = Latex_string + self.createLatexString(child)
		
			
		return Latex_string
		


	
		
		
class BBParser:
	def __init__(self):
		self.debugInt = 0
		print('initializing ...')
		
		self.symbol_manager = SymbolManager()


		self.latexgenerator = LatexGenerator()
		self.threshold_ratio_t = 0.9


		#debug:
		self.handling_file = ''
		

		# region_label
		# TL TR AB BL
		#

	def getDataFromFile(self): #debug
		#with open('Exp_train_giaidoan2.txt') as f:
		with open('ssd_train.txt') as f:
			z = f.readlines()
			self.test_candidate = z
		
		
	def debug(self):
		
		self.getDataFromFile()
		
		k = 0

		for file in self.test_candidate[:]:
			
			
			self.process(file)
			print('######### ' + str(k) +' ############')
			
			k += 1
			#try:
			#	self.process(file)
			#except:
			#	pass

	def process(self, input): #input is raw string
		raw_line = input.replace('\n', '').split(' ')
		
		self.handling_file = raw_line[0]
		raw_line = raw_line[1:]

		raw_line = list(map(int, raw_line))
			
		BB_List = []
			
		for i in range(raw_line[0]):
			BB_List.append(raw_line[5 * i + 1 : 5 * i + 6])
			
		self.preprocessingBBList(BB_List)
		BST = self.BuildBST(BB_List)
		print(self.handling_file)
		#self.debugPrint(BST)

		################ LEX ###########################################

		self.current_LBST = LBST(self.symbol_manager)


		LBSTTree = self.current_LBST.process(BST)
		

		latex_string = self.latexgenerator.process(LBSTTree)
		
		return latex_string
	
	def BuildBST(self, BB_list):
		BST = []
		if len(BB_list) == 0:
			return BST
		node_list = sorted(BB_list, key=itemgetter(0))
		
		retvalue = self.ExtractBaseline(node_list)
		
		return retvalue

	def debugPrint(self, tree):
		print(self.handling_file)
		for i in tree:
			sym, clas, align = self.symbol_manager.getSymbolFromIndex(i[4])
			#print(i)
			print(sym)
			for j in i[9]:
				print(j)
			print('---')
		
	def ExtractBaseline(self, rnode_list):
		if len(rnode_list) < 1:
			return rnode_list
		
		s_start = self.start(rnode_list)

		
		idx = 0
		for i in rnode_list:
			if i[0] == s_start[0] and i[1] == s_start[1] and i[4] == s_start[4]:
				break
			idx += 1
			
		del rnode_list[idx]

		
		baseline_symbols = self.Hor([s_start], rnode_list)

		
		updated_baseline = self.CollectRegion(baseline_symbols)


		for symbol in updated_baseline:
			for idx in range(len(symbol[self.Atb['child_temp']])):
				temp = self.ExtractBaseline(symbol[self.Atb['child_temp']][idx])
				
				#print('>>>>>')
				#print(symbol[self.Atb['child_temp']][idx])
				#print('---------')
				#print(temp)
				#print('<<<<<')
				symbol[self.Atb['child_temp']][idx] = temp				
				pass


		return updated_baseline

		
		
	def start(self, S_node_list): #return the index of first symbol in baseline
		if (len(S_node_list) < 1):
			return 0
		
		temp_list = S_node_list[:]
		while len(temp_list) > 1:
		
			sym_n = temp_list[-1]
			sym_n_1 = temp_list[-2]
			
			sym_n_sym, sym_n_class, sym_n_align = self.symbol_manager.getSymbolFromIndex(sym_n[4])
			
			if self.overlap(sym_n, sym_n_1) or self.Contains(sym_n, sym_n_1) or (sym_n_class == 'VariableRange' and not self.IsAdjacent(sym_n_1, sym_n)):
				#s_n dominate
				del temp_list[-2]
			else:
				del temp_list[-1]
				
		return temp_list[0]
				
		self.debugDraw(temp_list)
	
	def Hor(self, S_node_list_1, S_node_list_2):
		if len(S_node_list_2) == 0:
			return S_node_list_1
	
		current_symbol = S_node_list_1[-1]

		
		remaining_symbols, current_symbol_new = self.Partition(S_node_list_2, copy.deepcopy(current_symbol))

		
		#replace
		
		S_node_list_1.pop()
		
		S_node_list_1.append(current_symbol_new)
		
		

		if len(remaining_symbols) == 0:
			return S_node_list_1
		
		#6
		sym, clas, align = self.symbol_manager.getSymbolFromIndex(current_symbol_new[4])
		
		if clas == 'NonScripted':
			temp = self.start(remaining_symbols)
			temp = S_node_list_1 + [temp]
			
			return self.Hor(temp, remaining_symbols)
		
		SL = remaining_symbols[:]

		
		while len(SL) > 0:
			l1 = SL[0]

			
			if self.IsRegularHor(current_symbol_new, l1):
				####
				#if (len(SL) == 2):


				temp = self.CheckOverlap(l1, remaining_symbols)


				temp = S_node_list_1 + [temp] 
				


				return self.Hor(temp, remaining_symbols)


			SL = SL[1:]


		current_symbol_new = self.PartitionFinal(remaining_symbols, copy.deepcopy(current_symbol_new))
		
		temp = S_node_list_1[:]
		temp.pop()
		temp.append(current_symbol_new)
		
		return temp
		
	def CollectRegion(self, snode_list):
		temp = self.CollectRegionPartial(snode_list, 'TL')
		return self.CollectRegionPartial(temp, 'BL')

	def CollectRegionPartial(self, snode_list,region):
		if len(snode_list) == 0:
			return snode_list

		if region == 'TL':
			region_diagonal = 'TL'
			region_s1_diagonal = 'SUP'
			region_list = ['TL', 'T', 'SUP']
			region_vertical = 'T'
		else:
			region_diagonal = 'BL'
			region_s1_diagonal = 'SUB'
			region_list = ['BL', 'B', 'SUB']
			region_vertical = 'B'


		s1 = copy.deepcopy(snode_list[0])
		s1_new = copy.deepcopy(s1)

		snode_list_new = snode_list[1:]

		if len(snode_list) > 1:
			s2 = snode_list[1]
			superList, tleftList = self.PartitionSharedRegion(region_diagonal, s1, s2)
			s1_new = self.addRegion(region_s1_diagonal, superList, s1)
			s2_new = self.addRegion(region_diagonal, tleftList, self.removeRegion([region_diagonal], s2))

			del_idx = 0
			for i in snode_list_new:
				if i[0] == s2[0] and i[1] == s2[1] and i[4] == s2[4]:
					break
				del_idx = del_idx + 1 
			del snode_list[del_idx]

			snode_list.insert(del_idx, s2_new)

		syms1_new, class1_new, aligs1_new = self.symbol_manager.getSymbolFromIndex(s1_new[self.Atb['label']])
		if class1_new == 'VariableRange':
			#region_list = ['TL', 'T', 'SUP']

			s1_new = self.mergeRegion(region_list, region_vertical, s1)

		return [s1_new] + self.CollectRegion(snode_list_new)

	def IsRegularHor(self, snode1, snode2):
	
		sym1, clas1, alig1 = self.symbol_manager.getSymbolFromIndex(snode1[self.Atb['label']])
		sym2, clas2, alig2 = self.symbol_manager.getSymbolFromIndex(snode2[self.Atb['label']])
	
		cond_a = self.IsAdjacent(snode2, snode1)
		
		cond_b = snode1[1] > snode2[1] and snode1[3] < snode2[3] #1 in 2 horizontally
		cond_c = (sym2 == '(' or sym2 == ')') and snode2[1] < snode1[6] and snode2[3] > snode1[6]
		

		return cond_a or cond_b or cond_c
		
	def preprocessingBBList(self, BB_list): 
	#format: (tlx, tly, brx, bry, label, centroidx, centroidy, thres_sub, thres_sup, child)
	#child_temp: TL, BL, T, B, C, SUP, SUB
	#child_main: SUP, SUB, UPP, LOW
		self.Atb = {} #BB Attribute
		self.Atb['tlx'] = 0
		self.Atb['tly'] = 1
		self.Atb['brx'] = 2
		self.Atb['bry'] = 3
		self.Atb['label'] = 4
		self.Atb['centroidx'] = 5
		self.Atb['centroidy'] = 6
		self.Atb['thres_sub'] = 7
		self.Atb['thres_sup'] = 8
		self.Atb['child_temp'] = 9
		self.Atb['child_main'] = 10

		self.ChildLabel = {}
		self.ChildLabel['TL'] = 0
		self.ChildLabel['BL'] = 1
		self.ChildLabel['T'] = 2
		self.ChildLabel['B'] = 3
		self.ChildLabel['C'] = 4
		self.ChildLabel['SUP'] = 5
		self.ChildLabel['SUB'] = 6

		for BB in BB_list:
			sym, clas, align = self.symbol_manager.getSymbolFromIndex(BB[4])
			#centroidx
			if sym == '(':
				BB.append((BB[0] + BB[2]) / 2)
			elif sym == ')':
				BB.append((BB[0] + BB[2]) / 2)
			else:
				c = BB[0] + (BB[2] - BB[0]) / 2.0
				BB.append(c)
				
			#centroidy
			if align == 'Centred':
				c = BB[1] + (BB[3] - BB[1]) / 2.0
				BB.append(c)
			elif align == 'Ascender':
				c = BB[1] + (BB[3] - BB[1]) / 4.0 * 3
				BB.append(c)
			else:
				c = BB[1] + (BB[3] - BB[1]) / 4.0
				BB.append(c)
				
			#thres_sub
			height = BB[3] - BB[1] 
			
			if clas == 'NonScripted':
				BB.append(BB[6])
				BB.append(BB[6])
			#elif clas == 'Bracket':

			elif clas == 'Plain_Descender':
				BB.append(BB[1] + 0.5 * height + 0.5 * height * self.threshold_ratio_t)
				BB.append(BB[1] + height - 0.5 * height * self.threshold_ratio_t)
			else:
				BB.append(BB[1] + height * self.threshold_ratio_t)
				BB.append(BB[1] + height - height * self.threshold_ratio_t)
	
			BB.append([[], [], [], [], [], [], []])

			sym1, clas1, alig1 = self.symbol_manager.getSymbolFromIndex(BB[4])
			BB.append(sym1)

			#BB.append([[], [], [], []])
		
	def overlap(self, snode_1, snode_2): #Test whether snode_1 is a Nonscripted symbol that vertically overlaps snode_2.
	
		if (snode_1[0] == snode_2[0] and snode_1[1] == snode_2[1]):
			return False
	
		sym1, clas1, alig1 = self.symbol_manager.getSymbolFromIndex(snode_1[4])
		sym2, clas2, alig2 = self.symbol_manager.getSymbolFromIndex(snode_2[4])
		
		cond_b = clas1 == 'NonScripted'
		cond_c = snode_1[0] <= snode_2[5] and snode_2[5] < snode_1[2]
		cond_d = not self.Contains(snode_2, snode_1)
		
		cond_e_i = (sym2 == '(' or sym2 == ')') and (snode_2[1] <= snode_1[6] and snode_1[6] < snode_2[1]) and (snode_2[0] <= snode_1[5] and snode_1[5] < snode_2[2])
		
		cond_e_ii = (clas2 == 'NonScripted' or clas2 == 'VariableRange') and (snode_2[2] - snode_2[0] > snode_1[2] - snode_1[0])
		
		cond_e = (not cond_e_i) and (not cond_e_ii)
		
		ret = cond_b and cond_c and cond_d and cond_e
		return ret
		
	def CheckOverlap(self, snode, snode_list):
		longest = -1
		
		return_candidate = 0
		
		for node in snode_list:
			sym, clas, alig = self.symbol_manager.getSymbolFromIndex(node[4])
			if clas == 'NonScripted' and self.overlap(node, snode):
				w = node[2] - node[0]
				if w > longest:
					return_candidate = node
					longest = w

		if longest == -1:
			return snode
		else:
			return return_candidate
		
	def Contains(self, snode_1, snode_2): #for root
		sym1, clas1, alig1 = self.symbol_manager.getSymbolFromIndex(snode_1[4])
		sym2, clas2, alig2 = self.symbol_manager.getSymbolFromIndex(snode_2[4])
		
		cond_1 = clas1 == 'Root'
		cond_2 = snode_1[0] <= snode_2[5] and snode_2[5] < snode_1[2]
		cond_3 = snode_1[1] <= snode_2[6] and snode_2[6] < snode_1[4]
		return cond_1 and cond_2 and cond_3
		
		
	def IsAdjacent(self, snode_1, snode_2): #Test whether snode1 is horizontally adjacent to snode2, where snode1 may be to the left or right of snode2
		sym1, clas1, alig1 = self.symbol_manager.getSymbolFromIndex(snode_1[4])
		sym2, clas2, alig2 = self.symbol_manager.getSymbolFromIndex(snode_2[4])
		
		#format: (tlx, tly, brx, bry, label, centroidx, centroidy, thres_sub, thres_sup)
		#child: TL, BL, T, B, C
		#print('aa')
		#print(snode_1)
		#print(snode_2)
		#print('-----')
		#print(str(snode_2[7]) + ' > ' + str(snode_1[6]) )
		#print(str(snode_1[6]) + ' > ' + str(snode_2[8]) )

		return (clas2 != 'NonScripted') and (snode_2[7] > snode_1[6] and snode_1[6] > snode_2[8])
		
	def Partition(self, snode_list, snode):
		temp_list = snode_list[:]

		del_idx_list = []
		idx = 0



		for node in temp_list:	
			
			if node[5] < snode[0]: #centroidx_node < snode_minX

				if node[6] < snode[8]: #centroidy_node < snode_sup # Topleft
					snode[self.Atb['child_temp']][0].append(node[:])
					del_idx_list.append(idx)
				elif node[6] > snode[7]: #centroidy_node > snode_sub # Botleft
					snode[self.Atb['child_temp']][1].append(node[:])
					del_idx_list.append(idx)
			elif node[5] < snode[2]: #centroidx_node < snode_maxX

				if node[6] < snode[1]: #centroidy_node < snode_minY # Top
					snode[self.Atb['child_temp']][2].append(node[:])
					del_idx_list.append(idx)
				elif node[6] > snode[3]: #centroidy_node > snode_maxY # Bot
					snode[self.Atb['child_temp']][3].append(node[:])
					del_idx_list.append(idx)
				else: #Contain
					if node[0] != snode[0] or node[1] != snode[1] or node[4] != snode[4]:
						snode[self.Atb['child_temp']][4].append(node[:])
					del_idx_list.append(idx)
			else:
				if node[0] == snode[0] and node[1] == snode[1] and node[4] == snode[4]:
					del_idx_list.append(idx)
			idx += 1
		
		for i in del_idx_list[::-1]:
			del snode_list[i]
		
		return snode_list, snode
		
	def PartitionFinal(self, snode_list, snode):
		#format: (tlx, tly, brx, bry, label, centroidx, centroidy, thres_sub, thres_sup)
		#child: TL, BL, T, B, C, SUP, SUB
		for node in snode_list:
			if node[6] < snode[8]: #centroid < sup
				snode[self.Atb['child_temp']][5].append(node[:])
			else:
				snode[self.Atb['child_temp']][6].append(node[:])
				
		return snode
				
	def PartitionSharedRegion(self, region_label, snode1, snode2):

		S_node_list_1 = []
		S_node_list_2 = []

		idx = 0
		if region_label == 'BL':
			idx = 1

		rnode = SL = copy.deepcopy(snode2[self.Atb['child_temp']][idx])

		sym1, clas1, alig1 = self.symbol_manager.getSymbolFromIndex(snode1[4])
		sym2, clas2, alig2 = self.symbol_manager.getSymbolFromIndex(snode2[4])

		if clas1 == 'Nonscripted':
			S_node_list_1 = []
			return S_node_list_1, SL
		
		elif (clas2 != 'VariableRange') or (clas2 == 'VariableRange' and not self.HasNonEmptyRegion(snode2, 'T')):
			S_node_list_1 = SL
			return S_node_list_1, S_node_list_2

		elif clas2 == 'VariableRange' and self.HasNonEmptyRegion(snode2, 'T'):
			for i in SL:
				if self.IsAdjacent(i, snode2):
					S_node_list_1.append(i)
				else:
					S_node_list_2.append(i)

			return S_node_list_1, S_node_list_2

	def HasNonEmptyRegion(self, snode, region_label):
		#child: TL, BL, T, B, C, SUP, SUB

		return len(snode[self.Atb['child_temp']][self.ChildLabel[region_label]]) > 0

		#if region_label == 'TL':
		#	return len(snode[self.Atb['child_temp']][0]) > 0
		#elif region_label == 'BL'
		#	return len(snode[self.Atb['child_temp']][1]) > 0
		#elif region_label == 'T'
		#	return len(snode[self.Atb['child_temp']][2]) > 0
		#elif region_label == 'B'
		#	return len(snode[self.Atb['child_temp']][3]) > 0
		#elif region_label == 'C'
		#	return len(snode[self.Atb['child_temp']][4]) > 0
		#elif region_label == 'SUP'
		#	return len(snode[self.Atb['child_temp']][5]) > 0
		#elif region_label == 'SUB'
		#	return len(snode[self.Atb['child_temp']][6]) > 0
		#return False

	def debugDraw(self, temp_list):
		print('debug draw')
		sym1, clas1, alig1 = self.symbol_manager.getSymbolFromIndex(temp_list[0][4])


		img = Image.open('./hardimg/' + self.handling_file)
		

		fnt = ImageFont.truetype('./font/arial.ttf', 40)
		draw = ImageDraw.Draw(img)
		
		draw.rectangle(list(temp_list[0][:4]), outline='red')

		img.save('./result2/' + self.handling_file)
		
	def addRegion(self, region_label, list_to_add, snode):
		
		snode[self.Atb['child_temp']][self.ChildLabel[region_label]] = snode[self.Atb['child_temp']][self.ChildLabel[region_label]] + list_to_add

		#if region_label == 'TL':
		#	snode[self.Atb['child_temp']][0] = snode[self.Atb['child_temp']][0] + list_to_add
		#elif region_label == 'BL'
		#	snode[self.Atb['child_temp']][1] = snode[self.Atb['child_temp']][1] + list_to_add
		#elif region_label == 'T'
		#	snode[self.Atb['child_temp']][2] = snode[self.Atb['child_temp']][2] + list_to_add
		#elif region_label == 'B'
		#	snode[self.Atb['child_temp']][3] = snode[self.Atb['child_temp']][3] + list_to_add
		#elif region_label == 'C'
		#	snode[self.Atb['child_temp']][4] = snode[self.Atb['child_temp']][4] + list_to_add
		#elif region_label == 'SUP'
		#	snode[self.Atb['child_temp']][5] = snode[self.Atb['child_temp']][5] + list_to_add
		#elif region_label == 'SUB'
		#	snode[self.Atb['child_temp']][6] = snode[self.Atb['child_temp']][6] + list_to_add
		return snode

	def removeRegion(self, region_label_list, snode):
		for region_label in region_label_list:
			snode[self.Atb['child_temp']][self.ChildLabel[region_label]] = []
			#if region_label == 'TL':
			#	snode[self.Atb['child_temp']][0] = []
			#elif region_label == 'BL'
			#	snode[self.Atb['child_temp']][1] = []
			#elif region_label == 'T'
			#	snode[self.Atb['child_temp']][2] = []
			#elif region_label == 'B'
			#	snode[self.Atb['child_temp']][3] = []
			#elif region_label == 'C'
			#	snode[self.Atb['child_temp']][4] = []
			#elif region_label == 'SUP'
			#	snode[self.Atb['child_temp']][5] = []
			#elif region_label == 'SUB'
			#	snode[self.Atb['child_temp']][6] = []

		return snode

	def mergeRegion(self, region_label_list, region_label, snode):

		add_idx = self.ChildLabel[region_label]

		for region_to_merge in region_label_list:
			if region_to_merge == region_label:
				continue

			snode[self.Atb['child_temp']][add_idx] = snode[self.Atb['child_temp']][add_idx] + snode[self.Atb['child_temp']][self.ChildLabel[region_to_merge]]
			snode[self.Atb['child_temp']][self.ChildLabel[region_to_merge]] = []

		return snode

	##################################################################################
	##################################################################################
	##################################################################################
	##################################################################################




#obj = BBParser()
#obj.debug()
