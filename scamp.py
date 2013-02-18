# -*- coding: utf-8 -*-
# SCAMP -Symbian Cellular Automata Machine in Python
# (C) Giles R. Greenway, released under the GNU Public License v3
#15/04/2011

import appuifw, e32, os, os.path, graphics, random, re

try:
	import sysinfo
except:
	pass

try:
	begin_redraw()
	end_redraw()
except:
	def begin_redraw(junk1=0,junk2=0,junk3=0,junk4=0):
		pass

	def end_redraw(junk1=0,junk2=0,junk3=0,junk4=0):
		pass

def openurl(url):
	#if os.path.exists('z:\\system\\programs\\apprun.exe') and os.path.exists('z:\\System\\Apps\\Browser\\Browser.app'):
	#	e32.start_exe('z:\\system\\programs\\apprun.exe', 'z:\\System\\Apps\\Browser\\Browser.app'+ ' "%s"' %url,1)
	#else:
	e32.start_exe('BrowserNG.exe','4 "%s"' %url, 1)

class ProgressBar(object):
    # Implements a ProgressBar on Canvas
    def __init__(self, other_canvas, start=0, end=100,
                 color=(0,0,0), fill=(255,255,255),
                 outline=(0,0,0)):
        #canvas assignments
        self.canvas_copy = graphics.Image.new(other_canvas.size)
        self.canvas_copy.blit(other_canvas)
        self.return_canvas = graphics.Image.new(self.canvas_copy.size)
        self.return_canvas.blit(other_canvas)
        self.canvas = other_canvas
        #External box size
        self.box_w = int(self.canvas_copy.size[0] * 0.8)
        self.box_h = 50 #height of window
        self.box_l = int(self.canvas_copy.size[0] - self.box_w) / 2
        self.box_t = self.canvas_copy.size[1] - self.box_h - 5
        #ProgressBar size
        self.progr_margin_h = 5 #horizontal margins (left and right)
        self.prog_w = self.box_w - 2 * self.progr_margin_h
        self.prog_h = 18 #height of progressbar
        self.prog_l = self.box_l + self.progr_margin_h
        self.prog_t = self.box_t + int((self.box_h - self.prog_h) / 2)
        #internal progressbar expects that external has 1px border
        self.internal_w_max = self.prog_w - 2
        self.internal_h = self.prog_h - 2
        self.internal_l = self.prog_l + 1
        self.internal_t = self.prog_t + 1
        self.internal_w = 0
        #colors &amp; values
        self.start = start
        self.end = end
        self.value = start
        self.color = color
        self.outline = outline
        self.fill = fill
	self.rect = (self.box_l,self.box_t,self.box_l + self.box_w,self.box_t + self.box_h)
        #shows initial progressbar
        self.redraw()
 
    def close(self):
        #Closes the window and frees the image buffers memory
        self.canvas.blit(self.return_canvas)
        del self.canvas_copy
        del self.return_canvas
 
    def set_value(self, value):
        if value > self.end:
            value = self.end
        elif value < self.start:
            value = self.start
        self.value = value
        self.internal_w = int(((1.0 * self.value - self.start)/ \
                               (1.0 * self.end - self.start))   \
                              * self.internal_w_max)
        self.redraw()
 
    def redraw(self):
        """You don't need call redraw on application.
        Just use set_value to redraw the progressbar"""
        self.canvas_copy.blit(self.return_canvas)
        #external window
        self.canvas_copy.rectangle((self.box_l,
                                    self.box_t,
                                    self.box_l + self.box_w,
                                    self.box_t + self.box_h),
                                   outline=self.outline,
                                   fill=self.fill)
        #progressbar external border
        self.canvas_copy.rectangle((self.prog_l,
                                    self.prog_t,
                                    self.prog_l + self.prog_w,
                                    self.prog_t + self.prog_h),
                                   outline=self.outline,
                                   fill=self.fill)
        #progressbar core
        self.canvas_copy.rectangle((self.internal_l,
                                    self.internal_t,
                                    self.internal_l + self.internal_w,
                                    self.internal_t + self.internal_h),
                                   outline=None,
                                   fill=self.color)
        self.canvas.blit(self.canvas_copy)

##############################################################################
#						       			     #
# For a three-by-three array of binary cells,there are 2^9 combinations, and #
# thus 2^(2^9) possible CA rules.We can do a reasonable quick and cheap job  #
# of compressing such potentially huge integers by converting them to hex,   #
# (possibly) run-length encoding them, counting the number of different	     #
# hex-digits present to reduce the bit-count for each one as much as	     #
# possible, then encoding them in chunks of six bits using sixty-four	     #
# characters. Reasonably effective, if not entirely optimal.		     #
#									     #
##############################################################################

# 64 characters, each can represent 6 bits.
codechars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ@#'

# Quickly look up the value of each character.
codedict = dict()
for i, char in enumerate(codechars):
	codedict[char] = i

# Run-length encode a hex string.
def rle(hexstr):
	runchar = hexstr[0]
	run = 1	
	outstr = ""	
	for loop in range(1,len(hexstr)):
		if hexstr[loop] == runchar:
			if run < 17:
				run += 1
			else:
			# A run is represented by one repetition of the character, plus a hex digit that gives the length.	
				outstr += 2*runchar+"f"
				run = 1
		else:
			if run == 1:
			# A single character represents itself.	
				outstr += runchar
			else:
				outstr += 2*runchar+codechars[run-2]	
			runchar = hexstr[loop]	
			run = 1
	# Add on the final run.			
	if run == 1:
		outstr += runchar
	else:
		outstr += 2*runchar+codechars[run-2]
	return outstr

# Run-length decode a hex string.
def rld(codestr):
	char = 0
	outstr = ""
	while char<len(codestr)-1:
		if codestr[char] == codestr[char+1]:
		# Got a run, add it to the output and skip three chars.
			outstr += (2+codedict[codestr[char+2]])*codestr[char]
			char += 3
		else:
			outstr += codestr[char]
			char += 1
	if char < len(codestr):
	# Any left-overs must be a single char.
		outstr += codestr[char]

	return outstr

# Code a hex string using a set of 64 chars.
def crunch(x):
	chardict = dict()
	chars = 0
	present = 0
	# What hex digits are present in the string?
	for loop in range(16):
		if codechars[loop] in x:
			present = present | pow(2,loop)
			chardict[codechars[loop]] = chars
			chars += 1

	bits = 1
	# How many bits are needed per character in the string?
	while (pow(2,bits) < chars):
		bits += 1
	# Chop the hex string into six-bit chunks represented by a single character.
	outstr = ""
	thischar = 0
	charbits = 0
	for char in x:
		if charbits+bits <= 6:
		# Room for one more character?	
			thischar = (thischar << bits) + chardict[char]
			charbits += bits
		else:
		# Add the character to the ourput and start packing the next one.
			outstr += codechars[thischar]
			charbits = bits
			thischar = chardict[char]
	# Add the final char and work out how many decoded chars to throw away.
	thischar = thischar << (6-charbits)
	outstr += codechars[thischar]
	spare = (6-charbits)/bits
	# First three chars encode the hex digits present in the string.
	c3 = codechars[(present>>12)&63]
	c2 = codechars[(present>>6)&63]
	c1 = codechars[present&63]
			
	return c3+c2+c1+codechars[spare]+outstr

# Recover a "crunched" hex string.
def decrunch(x):
	# Recover the hex-digits present in the decoded string.
	present = (codedict[x[0]] & 63) << 12
	present += codedict[x[1]] << 6
	present += codedict[x[2]]
	spare = codedict[x[3]] 

	charlist = list()
	chars = 0
	# Reconstruct the list of hex digits.
	for loop in range(16):
		if pow(2,loop) & present:
			charlist.append(codechars[loop])
			chars += 1
	# Find the number of bits per hex digit.
	bits = 1
	while (pow(2,bits) < chars):
		bits += 1
	# Decode each char.
	outstr = ""
	mask = pow(2,bits) - 1
	for char in x[4:len(x)]:

		charval = codedict[char]
		charbits = 6

		while charbits >= bits:
			shift = (charbits/bits -1) * bits
			outstr += charlist[mask&(charval>>shift)]
			charbits -= bits

	return outstr[0:len(outstr)-spare]			

# Decide whether its worth doing the run-length encoding.
def squeeze(x):
	y = crunch(rle(x))
	z = crunch(x)
	# Set the sixth bit of the first char if we're bothering with the RLE.
	if len(y) <= len(z):
		return(codechars[codedict[y[0]]|32]+y[1:len(y)])
	else:
		return z

# Decode a coded string back to a hex string, with or without RLE.
def unsqueeze(x):
	firstchar = codedict[x[0]]
	if firstchar & 32:
		return rld(decrunch(x))
	else:
		return decrunch(x)

##############################################################################
#						       			     #
# Tokenize, parse and evaluate simple logical functions on the states of a   #
# cell and its neighbours.						     #
#									     #
##############################################################################

# Operators and their precedences: left shift, right shift, multiply, divide, modulo, bitwise AND, bitwise OR, bitwise XOR, logical AND, logical OR,
# logical XOR, greater than, less than, equal to, plus, minus and logical NOT.

operators = ("l",0),("r",0),("*",1),("/",1),("%",1),("&",2),("|",2),("$",2),("a",2),("o",2),("^",2),("x",2),(">",2),("<",2),("=",2),("+",3),("-",3),("!",4)

ops = ""
for index in range(len(operators)):
	ops += operators[index][0]

digits = "0123456789"
chars = "neswc"
brackets = "()"

legal = digits+chars+ops+brackets

ruletext = ""

# Various token-types:

class variable(object):
	def __init__(self,s,p):
		self.type = 2
		self.prior = 0
		self.pos = p
		self.label = s

class operator(object):
	def __init__(self,s,p):
		self.type = 3
		self.pos = p
		self.label = s
		for index in range(len(ops)):
			if s == ops[index]:
				self.prior = operators[index][1]

class constant(object):
	def __init__(self,s,p):
		self.type = 1
		self.prior = 0
		self.pos = p
		self.value = eval(s)

	def set_val(self,v):
		self.value = v

class bracket(object):
	def __init__(self,s,p):
		self.type = 4
		self.prior = 5
		self.pos = p	

		if s == "(":
			self.left = 1
		else:
			self.left = 0

class error_token(object):
	def __init__(self):
		self.type = 5
		self.prior = 999
		self.pos = 0
	
# Find the token-type of a character:	
def whatsit(x):
	for c in digits:
		if x == c:
			return 1
	for c in chars:
		if x == c:
			return 2
	for c in ops:
		if x == c:
			return 3
	for c in brackets:
		if x == c:
			return 4
	return 5

# Make a token from string s from position p of type t.
def make_token(s,t,p):
	if t == 1:
		return constant(s,p)
	if t == 2:
		return variable(s,p)
	if t == 3:
		return operator(s,p)
	if t == 4:
		return bracket(s,p)	
	if t == 5:
		return error_token()

# Tokenize string "s" given a list of possible variables. 
def tokenize(s,var_strings):
	token_list = list()
	# Get the first character.
	this_token = s[0]
	token_type = whatsit(this_token)
	token_pos = 0
	for index in range(1,len(s)):
		c = s[index]

		# What sort of token does the next char belong to?
		char_type = whatsit(c)

		if char_type > 2:
			# Must be an operator or a bracket. Store the current token and start a new one.
			token_list.append(make_token(this_token,token_type,token_pos))
			this_token = c
			token_type = char_type
			token_pos = index 
		else:
			if char_type == token_type:
				# The next char could be part of the current token,
				if token_type == 1:
					# Two consequtive digits, the number gets longer.
					this_token += c
				else:
					# Does the next char complete a two-char variable?
					trial = this_token + c
					if trial in var_strings:
						this_token = trial
					else:
						# Start a new token.
						token_list.append(make_token(this_token,token_type,token_pos))
						this_token = c
						token_type = char_type
						token_pos = index 																
			else:
				token_list.append(make_token(this_token,token_type,token_pos))
				this_token = c
				token_type = char_type
				token_pos = index


	token_list.append(make_token(this_token,token_type,token_pos))
	return token_list

# Shunting algorithim. Take the input string in_str and a list of valid chars, return a stack of tokens ready for evaluation.
def parse(in_str,valid):

	in_stack = list()
	out_stack = list()
	op_stack = list()

	in_stack = tokenize(in_str,valid)

	in_stack.reverse()

	while len(in_stack) > 0: # Iterate over the input stack.
		
		token = in_stack.pop()

		if token.type == 5: # Chuck back an error token and quit.
			out_stack = list()
			out_stack.append(token)
			return out_stack
		
		if token.type == 1 or token.type == 2: # Variables and constants go straight on the output stack.
			out_stack.append(token)

		if token.type == 3:

			if len(op_stack) > 0:
				while len(op_stack) > 0 and op_stack[len(op_stack)-1].prior <= token.prior:

					out_stack.append(op_stack.pop())

			op_stack.append(token)
				
		if token.type == 4:
			if token.left:
				op_stack.append(token) # Left brackets go to the operator stack.
			else:
				while len(op_stack) > 0: # Put the contents on the operator stack on the output stack 'till we find a left bracket. 
					op_token = op_stack.pop()
					if op_token.type == 3:
						out_stack.append(op_token)
					else:
						break
				
	while len(op_stack) > 0: # Push any leftover operators on the output stack.
		out_stack.append(op_stack.pop())

	return out_stack

def rp_eval(in_stack,vardict): # Evaluate a suitable stack of tokens, given the values of the variables.
	out_stack = list()
	
	for token in in_stack:

		if token.type == 1: # Contants go straight through.
			out_stack.append(token)
			
		if token.type == 2: # Push a constant with the varisble's value.
			out_stack.append(constant(str(vardict[token.label]),token.pos))
	
		if token.type == 3:

			toke = constant("0",0)	
			c2 = out_stack.pop().value # Take a value from the stack.

			if token.label != "!": 

				c1 = out_stack.pop().value # Binary operaters, take another value.
				
				if token.label == "+":
					toke.set_val(c1+c2)

				if token.label == "-":
					toke.set_val(c1-c2)

				if token.label == "*":
					toke.set_val(c1*c2)

				if token.label == "/":
					if c2 != 0:
						toke.set_val(int(c1/c2))
					else:
						toke.set_val(0)

				if token.label == "%": # Modular division
					if c2 != 0:
						toke.set_val(c1%c2)
					else:
						toke.set_val(0)
				
				if token.label == "&": # Bitwise AND
					toke.set_val(c1&c2)

				if token.label == "|": # Bitwise OR
					toke.set_val(c1|c2)

				if token.label == "$": # Bitwise XOE
					toke.set_val(c1^c2)

				if token.label == "^": # Raise to power
					toke.set_val(c1**c2)

				if token.label == "a": # Logical AND
					if c1 and c2:
						toke.set_val(1)
					else:
						toke.set_val(0)

				if token.label == "o": # Logical OR
					if c1 or c2:
						toke.set_val(1)
					else:
						toke.set_val(0)

				if token.label == "x": # Logical XOR
					if (c1 or c2) and not (c1 and c2):
						toke.set_val(1)
					else:
						toke.set_val(0)

				if token.label == ">": # Greater than
					if c1 > c2:
						toke.set_val(1)
					else:
						toke.set_val(0)

				if token.label == "<": # Less than
					if c1 > c2:
						toke.set_val(1)
					else:
						toke.set_val(0)

				if token.label == "=": # Equality
					if c1 == c2:
						toke.set_val(1)
					else:
						toke.set_val(0)
			else:
				if c2:
					toke.set_val(0)
				else:
					toke.set_val(1)

			out_stack.append(toke) # Put the result of the operation on the output stack.
	
	return out_stack.pop().value 

white = (255,255,255)
black = (0,0,0)
grey = (128,128,128)

red = (255,0,0)
yellow = (255,255,0)
green = (0,255,0)
blue = (0,0,255)
violet = (255,0,255)

class palette(object):
	def __init__(self,order):

		if order == 2:
			self.cols = [white,black]
		else:
			self.cols = order*[0]	
			n_cols = len(colours)
			spread = 256 / n_cols
			map = []
			big_list = 256*[0]

			for index in range(n_cols-1):
				map.append(index*spread)
				big_list[map[index]] = colours[index]
			map.append(255)
			big_list[255] = colours[n_cols-1]

			for index in range(n_cols-1):
				dc = 1+map[index+1]-map[index]
				dred = (colours[index+1][0]-colours[index][0])/dc
				dgreen = (colours[index+1][1]-colours[index][1])/dc
				dblue = (colours[index+1][2]-colours[index][2])/dc
				for col in range(map[index]+1,map[index+1]):
					big_list[col] = (self.setrgb(big_list[col-1][0]+dred),
					self.setrgb(big_list[col-1][1]+dgreen),
					self.setrgb(big_list[col-1][2]+dblue))

			for index in range(order-1):
				self.cols[index] = big_list[(index*255)/order]
			self.cols[order-1] = big_list[255]

			images[orientation].clear((self.cols[0],white,black)[bg_col])
			handle_redraw((0,0,w,h))

	def setrgb(self,rgb):
		if rgb > 255:
			return 255
		if rgb < 0:
			return 0
		return rgb
		
def midcol(col1,col2):
	return (int((col1[0]+col2[0])/2),int((col1[1]+col2[1])/2),
	int((col1[2]+col2[2])/2))

def s60sum(n):
	t = 0
	for x in n:
		t += x
	return t
		
class ca_par(object):
	def __init__(self,nm,vl,mi,mx):
		self.name = nm
		self.value = vl
		self.mini = mi
		self.maxi = mx

	def set_val(self,x):
		self.value = x

class ca_type(object):
	def __init__(self,prototype):

		lab,prs,eng,name,rec,seed,prog = prototype

		global ca_labels
		global ca_names		

		self.pars=[]
		for index in prs:
			self.pars.append(ca_par(index[0], index[1], index[2],index[3]))		
			
		self.values = {}

		self.reset()

		self.engine = eng
		self.name = name

		self.ca = 0		

		self.fields = []

		for index in self.pars:
			if index.name == u'history' or index.name == u"rule" or index.name == u'code':
				self.fields.append((index.name, 'text',unicode(index.value)))
			else:
				self.fields.append((index.name, 'number',index.value))

		self.pform = appuifw.Form(self.fields,
		appuifw.FFormEditModeOnly)
	        self.pform.save_hook = validate

		self.recordable = rec
		self.seed_type = seed
		self.prog  = prog

	def reset(self):
		for index in self.pars:
			self.values[index.name] = index.value

	def set_rule(self,newrule):

		if self.fields[0][0] == u'rule':
			self.values["rule"] = newrule

		if self.fields[0][0] == u'code':
			self.values["code"] = newrule

		self.fields = []				

		for index in self.pars:
			if index.name == u'history' or index.name == u'rule' or index.name == 'code':
				self.fields.append((index.name, 'text',unicode(self.values	[index.name])))
			else:
				self.fields.append((index.name, 'number',self.values[index.name]))
				
			self.pform = appuifw.Form(self.fields,appuifw.FFormEditModeOnly)	
	        	self.pform.save_hook = validate

	def draw(self):
		self.ca = self.engine(self.values)
		self.ca.draw()
		self.ca = 0

class ca_seed(object):
	def __init__(self,dm,lab,nm,prs):

		global seed_labels_1d
		global seed_names_1d
		global seed_labels_2d
		global seed_names_2d

		self.name = nm

		if dm == 1:
			self.dim = 1		
			seed_lables_1d.append(lab)
			seed_names_1d.append(nm)
		else:
			self.dim = 2
			seed_lables_2d.append(lab)
			seed_names_2d.append(nm)

		self.pars=[]
		for index in prs:
			self.pars.append(ca_par(
			index[0], index[1], index[2],index[3]))	

		self.values = {}

		self.reset()

		fields = []
		for index in self.pars:
			if index.maxi == -1 and index.mini == -1:
				fields.append((index.name, 'text',index.value))
			else:
				fields.append((index.name, 'number',index.value))
		self.pform = appuifw.Form(fields,
		appuifw.FFormEditModeOnly)
	        self.pform.save_hook = validate_seed

	def reset(self):
		for index in self.pars:
			self.values[index.name] = index.value

class ca_base(object):
	def __init__(self):
		self.ticktock = 0
		self.tocktick = 0
		self.then = 0
		self.now = 1

	def update(self):
		if click:
			savesnap()
		if rec and not self.ticktock % rec_inter:
			rec_img()
			self.tocktick = self.tocktick + 1
		self.ticktock = self.ticktock + 1

	def set_planes(self,n):
		self.planes = abs(n) + 1
		self.mask = (2 ** self.planes) - 1
		self.unmask = self.mask & (self.mask << 1)

		self.bitmasks = []
		for l in range(self.planes):
			self.bitmasks.append(2 ** l)

		if n >= 0:
			self.cellcol = self.state
			return palette(2 ** self.planes)
		else:
			 self.cellcol = self.change
			 return palette(1+self.planes)

	def swap(self):
		tmp = self.then
		self.then = self.now
		self.now = tmp

class ca_1d(ca_base):
	def __init__(self,siz):
		ca_base.__init__(self)
		self.size = siz
		self.cells = ([0]*self.size, [0]*self.size)
		self.states = 2

	def nearest(self,x):
		return [self.state((x+index)%self.size) for index in range(1,-2,-1)]

	def nextnearest(self,x):
		return [self.state((x+index)%self.size) for index in range(2,-3,-1)]

	def cells2int(self,x):
		return s60sum([(x[index] & 1)*pow(2,index) for index in range(len(x))])

	def change(self,n):
		cell = self.state(n)
		diff = self.planes
		for l in range(1,self.planes):
			if cell & self.bitmasks[l] != cell & cell & self.bitmasks[l-1]:
				diff = l-1
				break
		return diff

	def state(self,cell):
		return self.cells[self.then][cell]

	def next_state(self,cell):
		return self.cells[self.now][cell]

	def set_state(self,cell,val):
		self.cells[self.then][cell] = val

	def write(self,cell,val):
		self.cells[self.now][cell] = val

	def toggle(self,cell):
		self.cells[self.now][cell] = self.mask & ((self.cells[self.then][cell] << 1) | 1)

	def untoggle(self,cell):
		self.cells[self.now][cell] = (self.cells[self.then][cell] << 1) & self.unmask
	
	def clear(self):

		self.cells = [[0]*self.size, [0]*self.size]

	def seed(self):

		seed = one_d_seeds[chosen_seed]

		if seed.name == "singledefect":

			num = seed.values['number']
			just = seed.values['justify']		

			if seed.values['separation'] == 0:
				sep = w/(num+1)
			else:
				sep = seed.values['separation']

			if num == 1:
				sep = 0

			if just == 0:
				start = w/2 - ((num-1)*sep)/2
			if just == 1:
				start = 0
			if just == 2:
				start = w - (num-1)*sep - 1

			if start < 0:
				start = 0

			for index in range(num):
				self.set_state(start + (sep*index)%w,1)

		if seed.name == "altblocks":

			num = seed.values['number']
			on = seed.values['on']
			off = seed.values['off']

			if num > 0:
				run = (num * on) + ((num-1) * off)
				start = w/2 - run/2
				if start < 0:
					start = 0
				if run > w:
					run = w
			else:
				start = 0
				run = w

			on_run = 1
			run_count = 0

			for index in range(run):
				run_count = run_count + 1
				if on_run:
					self.set_state(start + index,1)
					if run_count == on:
						run_count = 0
						on_run = 0
				else:
					if run_count == off:
						run_count = 0
						on_run = 1

		if seed.name == "rand":

			prob = seed.values["probability"]
			width = seed.values["width"]
			if width == 0:
				start = 0
				end = w
			else:
				start = w/2 - width/2
				end = start + width

			for index in range(start,end):
				if random.choice(range(1000)) < prob:
					self.set_state(index,1)

		if seed.name == "bits":

			just = seed.values['justify']
			if just == 0:
				start = w/2 - 8
			if just == 1:
				start = 0
			if just == 2:
				start = w - 16
			
			bytes = [seed.values['left'],seed.values['right']]
			
			for byte in range(2):
				for bit in range(8):
					if bytes[byte] & 2**(7-bit):
						self.set_state(start+(8*byte)+bit,1)


	def draw(self):
		while drawing * finishing:
			for y in range(h):
				for x in range(w):
					plotpoint(x,y,self.pal.cols[self.cellcol(x)])

				for x in range(w):
					self.iterate(x)

				self.swap()
				handle_redraw((0,y,w,y))
				e32.ao_yield()
				if drawing != 1:
					break

			self.update()

class sim_1d(ca_1d):
	def __init__(self,par):

		self.pars = par
		ca_1d.__init__(self,w)
		self.seed()
		self.rules = [self.pars['rule']/pow(2,loop) % 2 for loop in range(8)]
		self.history = self.pars['history']
		self.pal = self.set_planes(self.history)

	def iterate(self,x):
		if self.rules[self.cells2int(self.nearest(x))]:
			self.toggle(x)
		else:
			self.untoggle(x)

def cells2int(x):
	return s60sum([(x[index] & 1)*pow(2,index) for index in range(len(x))])

def int2code(x):
	x = str(hex(x))
	return squeeze(x[2:len(x)-1])

def code2int(x):
	return eval('0x'+unsqueeze(x)+"L")

class nn_1d(ca_1d):
	def __init__(self,par):

		self.pars = par
		ca_1d.__init__(self,w)
		self.seed()
		self.rules = [self.pars['rule']/pow(2,loop) % 2 for loop in range(32)]
		self.bits = [pow(2,loop) for loop in range(32)]
		self.history = self.pars['history']
		self.pal = self.set_planes(self.history)

	def iterate(self,x):
		cells = self.nextnearest(x)

		if self.rules[self.cells2int(self.nextnearest(x))]:
			self.toggle(x)
		else:
			self.untoggle(x)

class pascal(ca_1d):
	def __init__(self,par):
		self.pars = par
		ca_1d.__init__(self,h)
		self.set_state(1,1)
		self.mod = self.pars['modulo']

		if self.pars['colours']:
			self.pal = palette(self.mod)
			self.paint = self.col_paint
		else:
			self.paint = self.mono_paint

	def mono_paint(self,off,x,y):
		if self.state(x) !=0:
			plotpoint(off+x,y,black)

	def col_paint(self,off,x,y):
		plotpoint(x+off,y,self.pal.cols[self.state(x)])

	def draw(self):
		centre = w/2
		for y in range(2,h-1):
			offset = centre - y/2
			for x in range(1,y+1):
				self.cells[self.now][x] = (self.state(x) +
				self.state(x-1)) % self.mod
				self.paint(offset,x,y)

			self.swap()
			handle_redraw((0,y,w,y))
			e32.ao_yield()
			if drawing != 1:
				break

class skew_pascal(ca_1d):

	def __init__(self,par):

		self.pars = par
		ca_1d.__init__(self,w)
		self.seed()		

		self.mod = self.pars['modulo']
		self.history = self.pars['history']
		self.colours = self.pars['colours']

		if self.colours:
			self.pal = palette(self.mod)
			self.iterate = self.col_paint
			self.cellcol = self.state
		else:
			self.modarray = (w*[0],w*[0])
			for x in range(w):
				self.modarray[self.then][x] = self.state(x)
			self.pal = self.set_planes(self.history)
			self.iterate = self.pix_paint

	def col_paint(self,x):
		self.write(x,(self.state(x)+self.state((x-1)%w)) % self.mod)

	def pix_paint(self,x):
		self.modarray[self.now][x] = (self.modarray[self.then][x]+self.modarray[self.then][(x-1)%w]) % self.mod
		if self.modarray[self.now][x]:
			self.toggle(x)
		else:
			self.untoggle(x)
		
class total_1d(ca_1d):
	def __init__(self,par):
		self.pars = par
		ca_1d.__init__(self,w)
		self.seed()
		self.order = self.pars['order']
		self.rule = self.pars['rule']
		self.pal = palette(self.order)
		self.cellcol = self.state
		self.states = self.order
		self.digits = []
		for index in range(3*self.order):
			power = pow(self.order,index)
			self.digits.append(int(self.rule/power) % self.order)

	def iterate(self,x):
		self.write(x,self.digits[s60sum(self.nearest(x))])

class cyclic_1d(ca_1d):
	def __init__(self,par):

		self.pars = par
		ca_1d.__init__(self,w)
		self.order = self.pars['order']
		if self.pars['neighbours']:
			self.delta = 3
		else:
			self.delta = 2
		self.pal = palette(self.order)
		self.cellcol = self.state
		self.map = []
		for x in range(w):
			self.set_state(x,random.choice(range(self.order)))
		for index in range(self.order):
			self.map.append((index+1)%self.order)

	def iterate(self,x):
		newstate = self.map[self.state(x)]
		self.write(x,self.state(x))				
		for dx in range(self.delta):
			if newstate == self.state((x-dx)%w) or newstate == self.state((x+dx)%w):
				self.write(x,newstate)
	
class ca_2d(ca_base):
	def __init__(self,neighbours,speed):
		ca_base.__init__(self)
		self.cells = [[],[]]
		self.states = 2
		self.neighbours = neighbours

		if neighbours:
			self.hood = self.nextnearest
		else:
			self.hood = self.nearest

		for index in range(w):
			self.cells[self.then].append(h*[0])
			self.cells[self.now].append(h*[0])

		self.init_state = self.set_state

		if speed:
			self.init_state = self.fast_set_state

			self.boxes = [[],[]]
			self.boxlist = [[],[]]

			self.boxwidth = 6
			self.boxheight = 6

			if w % self.boxwidth:
				self.xboxes = 1 + w/self.boxwidth
				self.lastboxwidth = w % self.boxwidth
			else:
				self.xboxes = w/self.boxwidth
				self.lastboxwidth = self.boxwidth

			if h % self.boxheight:
				self.yboxes = 1 + h/self.boxheight
				self.lastboxheight = h % self.boxheight
			else:
				self.yboxes = h/self.boxheight
				self.lastboxheight = self.boxheight

			for index in range(self.xboxes):
				self.boxes[self.then].append(self.yboxes*[0])
				self.boxes[self.now].append(self.yboxes*[0])

	def seed(self):

		if two_d_seeds[chosen_seed].name == "rand":

			prob = two_d_seeds[chosen_seed].values["probability"]

			progress = ProgressBar(images[orientation])

			for indx in range(w):
				for indy in range(h):
					if random.choice(range(1000)) < prob:
						self.init_state(indx,indy,1)

				grind = (indx*100)/w
				if indx % 5:
					progress.set_value(grind)
					handle_redraw(progress.rect)

			progress.close()
			del progress

		if two_d_seeds[chosen_seed].name == "singledefect":

			xnum = two_d_seeds[chosen_seed].values['xnumber']
			ynum = two_d_seeds[chosen_seed].values['ynumber']
		
			if two_d_seeds[chosen_seed].values['xsep'] == 0:
				xsep = w/(xnum+1)
			else:
				xsep = two_d_seeds[chosen_seed].values['xsep']

			if two_d_seeds[chosen_seed].values['ysep'] == 0:
				ysep = h/(ynum+1)
			else:
				ysep = two_d_seeds[chosen_seed].values['ysep']

			if xnum == 1:
				xsep = 0

			if ynum == 1:
				ysep = 0

			xstart = w/2 - ((xnum-1)*xsep)/2
			ystart = h/2 - ((ynum-1)*ysep)/2

			if xstart < 0:
				xstart = 0

			if ystart < 0:
				ystart = 0

			for xindex in range(xnum):
				x = xstart + (xsep*xindex)%w
				for yindex in range(ynum):
					self.init_state(x,ystart + (ysep*yindex)%h,1)

		if two_d_seeds[chosen_seed].name == "tabularasa":
			pass
		
		if two_d_seeds[chosen_seed].name == "gradient":
			x0 = two_d_seeds[chosen_seed].values['x0']
			x1 = two_d_seeds[chosen_seed].values['x1']
			y0 = two_d_seeds[chosen_seed].values['y0']
			y1 = two_d_seeds[chosen_seed].values['y1']
			
			xc = w/2
			yc = h/2
			
			dx = (x1-x0) / xc
			dy = (y1-y0) / yc
			
			for x in range(w):
				xprob = x0 + dx*abs(xc - x)
				for y in range(h):
					yprob = y0 + dy*abs(yc - y)
					if random.choice(range(1000)) < xprob and random.choice(range(1000)) < yprob:
						self.init_state(x,y,1)

		if two_d_seeds[chosen_seed].name == "bitmap":
			mapcols = two_d_seeds[chosen_seed].values['cols']
			mapwidth = 4 * mapcols
			xjust = two_d_seeds[chosen_seed].values['xjust'] 
			if xjust == 0:
				xstart = w/2 - mapwidth/2
			if xjust == 1:
				xstart = 0
			if xjust == 2:
				xstart = w - mapwidth
			mapchars = two_d_seeds[chosen_seed].values['bits']
			mapheight = len(mapchars) / mapcols 
			yjust = two_d_seeds[chosen_seed].values['yjust'] 
			if yjust == 0:
				ystart = h/2 - mapheight/2
			if yjust == 1:
				ystart = 0
			if yjust == 2:
				ystart = h - mapheight

			for y in range(mapheight):
				charpos = y*mapcols
				rowchars = mapchars[charpos:charpos+mapcols]
				rowpixels = []
				for char in rowchars:
					charval = eval('0x'+char)
					rowpixels += [ (charval & bit) <> 0 for bit in [8,4,2,1] ]
				for x in range(mapwidth):
					if rowpixels[x]:
						self.init_state(xstart+x,ystart+y,1)
				

	def change(self,x,y):
		cell = self.state(x,y)
		diff = self.planes
		for l in range(1,self.planes):
			if cell & self.bitmasks[l] != cell & cell & self.bitmasks[l-1]:
				diff = l-1
				break
		return diff

	def state(self,x,y):
		return self.cells[self.then][x][y]

	def set_state(self,x,y,val):
		self.cells[self.then][x][y] = val
	
	def initbox(self,x,y):

		if self.boxes[self.then][x][y] != 1:
			self.boxes[self.then][x][y] = 1
			self.boxlist[self.then].append((x,y))

	def addbox(self,x,y):

		if self.boxes[self.now][x][y] != 1:
			self.boxes[self.now][x][y] = 1
			self.boxlist[self.now].append((x,y))

	def fast_set_state(self,x,y,val):
		self.cells[self.then][x][y] = val

		xb = x/self.boxwidth
		yb = y/self.boxheight

		if xb == self.xboxes - 1:
			bw = self.lastboxwidth
		else:
			bw = self.boxwidth

		if yb == self.yboxes - 1:
			bh = self.lastboxheight
		else:
			bh = self.boxheight

		self.initbox(xb,yb)

		left = right = top = bottom = 0

		if x == xb * self.boxwidth:
			left = 1
		if x == xb*self.boxwidth + bw - 1:
			right = 1
		if y == yb * self.boxheight:
			top = 1
		if y == yb * self.boxheight + bh - 1:
			bottom = 1

		if left:
			xn = (xb-1) % self.xboxes
		if right:
			xn = (xb+1) % self.xboxes
		if top:
			yn = (yb-1) % self.yboxes
		if bottom:
			yn = (yb+1) % self.yboxes

		if left or right:
			self.initbox(xn,yb)

		if top or bottom:
			self.initbox(xb,yn)

		if self.neighbours and (left or right) and (top or bottom):
			self.initbox(xn,yn)

	def write(self,x,y,val):
		self.cells[self.now][x][y] = val

	def toggle(self,x,y):
		self.cells[self.now][x][y] = self.mask & ((self.cells[self.then][x][y] << 1) | 1)

	def untoggle(self,x,y):
		self.cells[self.now][x][y] = (self.cells[self.then][x][y] << 1) & self.unmask

	def turnon(self,x,y):
		self.cells[self.then][x][y] = self.mask & ((self.cells[self.then][x][y] << 1) | 1)

	def turnoff(self,x,y):
		self.cells[self.then][x][y] = (self.cells[self.then][x][y] << 1) & self.unmask

	def nearest(self,x,y):
		return [self.state(x,(y+1)%h), self.state((x+1)%w,y),
		self.state(x,(y-1)%h), self.state((x-1)%w,y)]

	def cnearest(self,x,y):
		return [self.state(x,y),self.state(x,(y-1)%h), self.state((x+1)%w,y),self.state(x,(y+1)%h), self.state((x-1)%w,y)]

	def nextnearest(self,x,y):
		return [self.state(x,(y+1)%h), self.state((x+1)%w,y),
		self.state(x,(y-1)%h), self.state((x-1)%w,y),
		self.state((x-1)%w,(y-1)%h),
		self.state((x+1)%w,(y-1)%h),
		self.state((x+1)%w,(y+1)%h),
		self.state((x-1)%w,(y+1)%h)]

	def cnnearest(self,x,y):
		return [self.state(x,y),self.state(x,(y-1)%h),
		self.state((x+1)%w,(y-1)%h),
		self.state((x+1)%w,y),
		self.state((x+1)%w,(y+1)%h),
		self.state(x,(y+1)%h),
		self.state((x-1)%w,(y+1)%h),
		self.state((x-1)%w,y),
		self.state((x-1)%w,(y-1)%h)]

	def rseed(self,threshold):
		progress = ProgressBar(images[orientation])
		for xx in range(w):
			for yy in range(h):
				if random.random() < threshold:
					self.init_state(xx,yy,1)
				else:
					self.init_state(xx,yy,0)
			grind = (xx*100)/w
			progress.set_value(grind)
			handle_redraw(progress.rect)
		progress.close()
		del progress

	def rfill(self,ord):
		progress = ProgressBar(images[orientation])
		for xx in range(w):
			for yy in range(h):
				self.set_state(xx,yy,random.choice(range(ord)))
			grind = (xx*100)/w
			progress.set_value(grind)
			handle_redraw(progress.rect)
		progress.close()
		del progress

	def scan(self):

		while drawing * finishing:

			for y in range(h):
			
				for x in range(w):			
					self.paint_cell(x,y)					

				redraw((0,y,w,y))
				e32.ao_yield()
				if drawing != 1:
					break

			self.swap()
			self.update()

	def changed(self,x,y):
		self.paint_cell(x,y)
		if self.cells[self.now][x][y] == self.cells[self.then][x][y]:
			return 0
		else:
			return 1

	def trace(self):

		dxdy = {'left':(-1,0), 'right':(1,0),
			'top':(0,-1), 'bottom':(0,1),
			'topleft':(-1,-1), 'topright':(1,-1),
			'bottomleft':(-1,1), 'bottomright':(1,1)}

		if self.neighbours:
			maxdir = 8
		else:
			maxdir = 4

		while drawing * finishing and len(self.boxlist[self.then]):

			self.boxlist[self.now] = []
			self.boxes[self.now] = []
			for index in range(self.xboxes):
				self.boxes[self.now].append(self.yboxes*[0])
	
			for box in self.boxlist[self.then]:
				direct = {'left':0, 'right':0, 'top':0, 'bottom':0,
				'topleft':0, 'topright':0, 'bottomleft':0, 'bottomright':0}
							
				sides = 0
				interior = 0
				bx,by = box
		
				if bx == self.xboxes - 1:
					boxw = self.lastboxwidth
				else:
					boxw = self.boxwidth

				if by == self.yboxes - 1:
					boxh = self.lastboxheight
				else:
					boxh = self.boxheight

				boxleft = self.boxwidth * bx
				boxtop = self.boxheight * by
				boxright = boxleft + boxw - 1
				boxbottom = boxtop + boxh - 1
			 	
				redraw((boxleft,boxtop,boxright,boxbottom))
				e32.ao_yield()	

				if self.changed(boxleft,boxtop):
					direct['left'] = direct['top'] = direct['topleft'] = 1
				if self.changed(boxright,boxtop):
					direct['right'] = direct['top'] = direct['topright'] = 1
				if self.changed(boxleft,boxbottom):
					direct['left'] = direct['bottom'] = direct['bottomleft'] = 1
				if self.changed(boxright,boxbottom):
					direct['right'] = direct['bottom'] = direct['bottomright'] = 1

				for x in range(boxleft+1,boxright):
					direct['top'] += self.changed(x,boxtop)
				for x in range(boxleft+1,boxright):
					direct['bottom'] += self.changed(x,boxbottom)
				for y in range(boxtop+1,boxbottom):
					direct['left'] += self.changed(boxleft,y)
				for y in range(boxtop+1,boxbottom):
					direct['right'] += self.changed(boxright,y)
				
				for y in range(boxtop+1,boxbottom):
					for x in range(boxleft+1,boxright):
						interior += self.changed(x,y)

				dirs = direct.values()

				for key in direct.iterkeys():
					if direct[key]:
						sides = 1
						dx,dy = dxdy[key]
						self.addbox((bx+dx)%self.xboxes,(by+dy)%self.yboxes)


				if interior or sides:
					self.addbox(bx,by)

				if drawing != 1:
					break

			self.swap()
			self.update()

	def draw(self):
		if self.speed:
			self.trace()
		else:
			self.scan()

class sim_2d(ca_2d):
	def __init__(self,par):
		self.pars = par
		self.rules = [self.pars['rule']/pow(2,loop) % 2 for loop in range(32)]
		self.history = self.pars['history']
		self.speed = self.pars['speed']
		self.pal = self.set_planes(self.history)

		ca_2d.__init__(self,0,self.speed)
		self.pal = self.set_planes(self.history)
		self.seed()

	def cells2int(self,x):
		return s60sum([(x[index] & 1)*pow(2,index) for index in range(len(x))])

	def paint_cell(self,x,y):
		plotpoint(x,y,self.pal.cols[self.cellcol(x,y)])

		if self.rules[cells2int(self.cnearest(x,y))]:
			self.toggle(x,y)
		else:
			self.untoggle(x,y)

class nn_sim_2d(ca_2d):
	def __init__(self,par):
		self.pars = par

		rule = code2int(self.pars['code'])

		self.rules = [rule/pow(2,loop) % 2 for loop in range(512)]
		self.history = self.pars['history']
		self.speed = self.pars['speed']

		ca_2d.__init__(self,1,self.speed)
		self.pal = self.set_planes(self.history)
		self.seed()

	def paint_cell(self,x,y):
		plotpoint(x,y,self.pal.cols[self.cellcol(x,y)])

		if self.rules[cells2int(self.cnnearest(x,y))]:
			self.toggle(x,y)
		else:
			self.untoggle(x,y)

class cyc_demon(ca_2d):
	def __init__(self,par):
		self.pars = par
		ca_2d.__init__(self,self.pars['neighbours'],0)
		self.order = self.pars['order']
		self.states = self.order
		self.pal = palette(self.order)
		self.map = []
		for index in range(self.order):
			self.map.append((index+1)%self.order)
		self.rfill(self.order)

	def draw(self):
		self.scan()		

	def paint_cell(self,x,y):
		this_state = self.state(x,y)
		plotpoint(x,y,self.pal.cols[this_state])
		self.write(x,y,this_state)
		state = self.map[this_state]
		for cell in self.hood(x,y):
			if cell == state:
				self.write(x,y,state)
				break
			
class langton(ca_2d):
	def __init__(self,par):
		self.pars = par
		ca_2d.__init__(self,par['neighbours'],0)
		self.seed()
		ant_num = par['number']
		self.order = par['order']
		self.dt = par['dt']
		self.states = self.order
		self.pal = palette(self.order)
		self.rules = [ant_num/pow(2,loop) % 2 for loop in
		range(self.order)]
		if par['neighbours']:
			self.dx = [0,1,1,1,0,-1,-1,-1]
			self.dy = [-1,-1,0,1,1,1,0,-1]
			self.dirs = 8
		else:
			self.dx = [0,1,0,-1]
			self.dy = [-1,0,1,0]
			self.dirs = 4

	def draw(self):
		x = w/2
		y = h/2
		xmi = x
		xma = x
		ymi = y
		yma = y		
		head = 0
		while drawing * finishing:
			for junk in range(self.dt):
				this_state = self.state(x,y)
				if self.rules[this_state]:
					head = (head+1)	% self.dirs
				else:
					head = (head-1) % self.dirs
				this_state = (this_state+1) % self.order
				self.set_state(x,y,this_state)
				plotpoint(x,y,self.pal.cols[this_state])
				x = (x + self.dx[head]) % w
				y = (y + self.dy[head]) % h

				if x > xma:
					xma = x
				if x < xmi:
					xmi = x
				if y > yma:
					yma = y
				if y < ymi:
					ymi = y

			handle_redraw((0,y,w,y))

			self.update()
		
			e32.ao_yield()

class turmites(ca_2d):
	def __init__(self,par):
		ca_2d.__init__(self,par['neighbours'],0)
		self.history = par['history']
		self.pal = self.set_planes(self.history)		
		self.dt = par['dt']

		if par['neighbours']:
			self.dx = [0,1,1,1,0,-1,-1,-1]
			self.dy = [-1,-1,0,1,1,1,0,-1]
			self.dirs = 8
		else:
			self.dx = [0,1,0,-1]
			self.dy = [-1,0,1,0]
			self.dirs = 4
		
		rule = par['rule']

		self.statetab = [[0,0],[0,0]]
		self.tile = [[0,0],[0,0]]
		self.rotate = [[0,0],[0,0]]

		changedir = [0,-1,1,self.dirs/2]

		for t in range(2):
			for s in range(2):
				byte = (rule >> (8*(1 - s))) & 255
				nibble = (byte >> (4*(1 - t))) & 15

				self.statetab[t][s] = nibble & 1
				if nibble & 8:
					self.tile[t][s] = 1
				else:
					self.tile[t][s] = 0
				self.rotate[t][s] = changedir[(nibble>>1)&3]

	def draw(self):
		x = w/2
		y = h/2
		xmi = x
		xma = x
		ymi = y
		yma = y
		head = 0
		this_state = 0
		while drawing * finishing:
			for junk in range(self.dt):	
				this_tile = self.state(x,y) & 1

				head = (head + self.rotate[this_tile][this_state]) % self.dirs

				if self.tile[this_tile][this_state]:
					self.turnon(x,y)
				else:
					self.turnoff(x,y)

				plotpoint(x,y,self.pal.cols[self.cellcol(x,y)])

				this_state = self.statetab[this_tile][this_state]

				x = (x + self.dx[head]) % w
				y = (y + self.dy[head]) % h

				if x > xma:
					xma = x
				if x < xmi:
					xmi = x
				if y > yma:
					yma = y
				if y < ymi:
					ymi = y

			handle_redraw((xmi,ymi,xma,yma))
			self.update()
		
			e32.ao_yield()

class bbrain(ca_2d):
	def __init__(self,par):
		self.speed = par['speed']
		ca_2d.__init__(self,par['neighbours'],self.speed)
		self.cols = [black,white,red]
		self.rseed(par['occupancy']/1000.0)

	def paint_cell(self,x,y):
		this_state = self.state(x,y)
		plotpoint(x,y,self.cols[this_state])
		if this_state == 0:
			total = 0
			for cell in self.hood(x,y):
				total += cell & 1
				if total == 2:
					self.write(x,y,1)
				else:
					self.write(x,y,0)
		else:
			if this_state == 1:
				self.write(x,y,2)
			else:
				self.write(x,y,0)

class bzr(ca_2d):
	def __init__(self,par):

		self.pars = par
		self.order = self.pars['order']
		ca_2d.__init__(self,par['neighbours'],0)
		self.n = self.order-1;
		self.pal = palette(self.order)
		self.states = self.order
		self.k1 = self.pars['k1']
		self.k2 = self.pars['k2']
		self.g = self.pars['g']

		self.pal = palette(self.order)
		self.rfill(self.order)

	def draw(self):
		self.scan()

	def paint_cell(self,x,y):
		this_state = self.state(x,y)
		plotpoint(x,y,self.pal.cols[this_state])

		s = this_state

		if s > 0:
			a = 1
		else:
			a=0

		b = 0
					
		for cell in self.hood(x,y):
			s = s + cell
			if cell > 0:
				if cell < self.n:
					a = a + 1
				else:
					b = b + 1
				
		if this_state == 0:
			newval = int(a/self.k1) + int(b/self.k2)

		if this_state > 0 and this_state < self.n:
			newval = int(s/a) + self.g

		if this_state == self.n:
			newval = 0

		if newval > self.n:
			newval = self.n

		self.write(x,y,newval)

class conlife(ca_2d):
	def __init__(self,par):

		self.pars = par

		self.speed = self.pars['speed']
		ca_2d.__init__(self,1,self.speed)

		self.rseed(self.pars['occupancy']/1000.0)
		self.history = self.pars['history']
		self.pal = self.set_planes(self.history)
		self.living = [0,0,1,1,0,0,0,0,0]
		self.dead = [0,0,0,1,0,0,0,0,0]
		self.livedead = [self.dead,self.living]

	def paint_cell(self,x,y):
		this_cell = self.state(x,y)
					
		plotpoint(x,y,self.pal.cols[self.cellcol(x,y)])
		total = 0

		for cell in self.hood(x,y):
			if cell & 1:
				total += 1

		if self.livedead[this_cell & 1][total]:
			self.toggle(x,y)
		else:
			self.untoggle(x,y)

class total_2d(ca_2d):

	def __init__(self,par):

		self.pars = par

		self.d_rule = self.pars['d_rule']
		self.l_rule = self.pars['l_rule']
		self.neighbours = self.pars['neighbours']
		self.speed = self.pars['speed']

		ca_2d.__init__(self,self.pars['neighbours'],self.speed)
		self.seed()		

		self.history = self.pars['history']
		self.pal = self.set_planes(self.history)

		if self.neighbours == 0:
			self.d_rule &= 31
			self.l_rule &= 31

		self.bits = [pow(2,n) for n in range(9)]

	def paint_cell(self,x,y):

		this_cell = self.state(x,y)	

		plotpoint(x,y,self.pal.cols[self.cellcol(x,y)])

		total = 0

		for cell in self.hood(x,y):
			if cell & 1:
				total += 1

		if this_cell & 1:
			livedead = self.l_rule & self.bits[total]
		else:
			livedead = self.d_rule & self.bits[total]

		if livedead:

			self.toggle(x,y)
		else:

			self.untoggle(x,y)

def set_pal():
	global colours
	try:
		colours = pal_list[int(appuifw.popup_menu(pal_lables, u"Select:"))][1]
	except:
		pass

def set_bg_col():
	global bg_col

	bg_col = appuifw.popup_menu([u"Default",u"White",u"Black"],u"Choose colour:")

def automata():
	global chosen_ca,ca_proto,automaton
	try:
		chosen_ca = int(appuifw.popup_menu(ca_lables, u"Select:"))
		automaton = ca_type(ca_proto[chosen_ca])
		main_menu()
	except:
		pass
	
def param():

	automaton.pform.execute()
	e32.ao_yield()	

def validate(state):
	for index in range(len(state)):
		if automaton.pars[index].name != u"code":
			try:
				if int(state[index][2]) > automaton.pars[index].maxi:
					e32.ao_yield()	
					return False
				if int(state[index][2]) < automaton.pars[index].mini:
					e32.ao_yield()	
					return False
			except:
				return False

	for index in range(len(state)):
		if automaton.pars[index].name != u"code":
			automaton.values[automaton.pars[index].name] = int(state[index][2])
		else:
			automaton.values[automaton.pars[index].name] = state[index][2]

	e32.ao_yield()	
	return True

def validate_seed(state):
	global automaton

	if automaton.seed_type == 1:
		par_list = one_d_seeds[chosen_seed].pars

	if automaton.seed_type == 2:
		par_list = two_d_seeds[chosen_seed].pars

	for index in range(len(state)):
		if par_list[index].mini == -1 and par_list[index].maxi == -1:
			for char in str(state[index][2]):
				if char.lower() not in '0123456789abcdef':
					e32.ao_yield()	
					return False
			continue
		
		if int(state[index][2]) > par_list[index].maxi:
			e32.ao_yield()	
			return False
		if int(state[index][2]) < par_list[index].mini:
			e32.ao_yield()	
			return False

	if automaton.seed_type == 1:
		for index in range(len(state)):
			one_d_seeds[chosen_seed].values[one_d_seeds[chosen_seed].pars[index].name] = int(state[index][2])

	if automaton.seed_type == 2:
		for index in range(len(state)):
			if par_list[index].mini == -1 and par_list[index].maxi == -1:
				two_d_seeds[chosen_seed].values[two_d_seeds[chosen_seed].pars[index].name] = state[index][2]
			else:	
				two_d_seeds[chosen_seed].values[two_d_seeds[chosen_seed].pars[index].name] = int(state[index][2])

	e32.ao_yield()	
	return True

def img_name():
	name = ca_names[chosen_ca]
	for index in automaton.pars:
		name += "_"+str(automaton.values[index.name])
	return name 

def img_num():
	hexstr = str(hex(automaton.ca.tocktick))
	numstr = hexstr[2:]
	zeroes = 4-len(numstr)
	return zeroes*"0"+numstr

def hiresplot(x,y,c):
	images[orientation].point((x,y),outline = c,width = 1)

def lowresplot(x,y,c):
	global box,xoff,yoff
	xl = xoff + box * x
	yt = yoff + box * y
	images[orientation].rectangle((xl,yt,xl+box,yt+box),outline = c, fill = c,width = 1)
	
def seed():
	global chosen_seed

	if automaton.seed_type == 1:
		try:
			chosen_seed = int(appuifw.popup_menu(seed_lables_1d, u"Select:"))
		except:
			pass
	else:
		try:
			chosen_seed = int(appuifw.popup_menu(seed_lables_2d, u"Select:"))
		except:
			pass
def seedparam():
		if automaton.seed_type == 1:
			one_d_seeds[chosen_seed].pform.execute()
		if automaton.seed_type == 2:
			two_d_seeds[chosen_seed].pform.execute()
		e32.ao_yield()	

def save():
	images[orientation].save(fname, bpp=24, compression="fast")

def draw():
	global c_wid,c_hi,imgages,drawing,finishing,click,rec,fname

	c_wid, c_hi = canvas.size
	images[orientation] = graphics.Image.new((c_wid, c_hi))
	find_res()
	set_res()
	
	fname = img_path+u"\\"+img_name()+".png"

	if automaton.recordable:
		appuifw.app.menu = [(u"Snapshot", snap),(u"Start Recording",start_rec), (u"Finish", finish),
	(u"Stop",stop)]
	else:
		appuifw.app.menu = [(u"Stop", stop)]

	e32.ao_yield() 
	finishing = 1
	drawing = 1
	rec = 0
	click = 0

	automaton.draw()
	drawing = 0
	finishing = 0
	main_menu()

def snap():
	global click
	click = 1

def stop():
	global drawing
	drawing = 0

def finish():
	global finishing
	finishing = 0

def savesnap():
	global click
	click = 0
	snapname = img_path+u"\\"+img_name()+"_"+img_num()+".png"
	images[orientation].save(snapname, bpp=24, compression="no")

def start_rec():
	global rec,rec_path,rec_inter
	rec = 1
	rec_path = img_path + u"\\" + img_name()
	if not os.path.exists(rec_path):
		os.makedirs(rec_path)

	rec_inter = appuifw.query(u"Recording interval?", "number")

	if rec_inter < 1:
		rec_inter = 1	

	appuifw.app.menu = [(u"Snapshot", snap),(u"Stop Recording",stop_rec),
	(u"Finish", finish),(u"Stop",stop)]

def draw_rec():

	global c_wid,c_hi,images,drawing,finishing,click,rec,fname,types,rec,rec_path,rec_inter

	c_wid, c_hi = canvas.size
	images[orientation] = graphics.Image.new((c_wid, c_hi))
	find_res()
	set_res()

	rec_path = img_path + u"\\" + img_name()
	if not os.path.exists(rec_path):
		os.makedirs(rec_path)
	
	fname = img_path+u"\\"+img_name()+".png"

	if automaton.recordable:
		rec = 1
		appuifw.app.menu = [(u"Snapshot", snap),(u"Stop Recording",stop_rec),
		(u"Finish", finish),(u"Stop",stop)]
		rec_inter = appuifw.query(u"Recording interval?", "number")		
	else:
		appuifw.app.menu = [(u"Stop", stop)]

	if rec_inter < 1:
		rec_inter = 1

	e32.ao_yield() 
	finishing = 1
	drawing = 1
	click = 0
	automaton.draw()

	drawing = 0
	finishing = 0
	main_menu()

def rec_img():
	name = rec_path+u"\\"+img_num()+".png"
	images[orientation].save(name, bpp=24, compression="no")

def stop_rec():
	global rec
	rec = 0
	appuifw.app.menu = [(u"Snapshot", snap),(u"Start Recording",start_rec),
	(u"Finish", finish),(u"Stop",stop)]

def handle_redraw(rect):
	if images[orientation]:
		begin_redraw(rect[0],rect[1],rect[2],rect[3])
		canvas.blit(images[orientation],target=(rect[0],rect[1],rect[2],rect[3]),
		source=((rect[0],rect[1],rect[2],rect[3]),images[orientation].size),scale=0)
		end_redraw()

def low_res_redraw(rect):
	left = xoff + box*rect[0]
	top = yoff + box*rect[1]
	right = xoff + box*rect[2]
	bottom = yoff + box*rect[3]
	begin_redraw(left,top,right,bottom)
	canvas.blit(images[orientation],target=(left,top,right,bottom),source=((left,top,right,bottom),
	images[orientation].size),scale=0)
	end_redraw()

def handle_event(junk):
	pass

def handle_resize(rect):
	global c_wid,c_hi,orientation

	if drawing == 0:
		try:
			c_wid, c_hi = canvas.size
			if c_hi > c_wid:
				orientation = 1
			else:
				orientation = 0

			find_res()
			set_res()

			canvas.blit(imgages[orientation],target=(rect[0],rect[1],rect[2],rect[3]),source=((0,0,c_wid,c_hi),imgages[orientation].size),scale=0)
		except:
			pass

def quit():
	global drawing
	drawing = 0
	app_lock.signal()

def img_dir():
	global img_path

	choice = appuifw.popup_menu(imgdirs,u"Save to:")
	try:
		img_path = imgdirs[choice]
	except:
		pass

def find_res():
	global res,res_lables,xoff,yoff

	wi = c_wid
	hi = c_hi
	res = []
	res_lables = []
	d = 1

	for n in range(4):

		res.append((wi,hi,d,xoff,yoff))
		res_lables.append(unicode(wi)+u" x "+unicode(hi))

		wi = int(wi/2)
		hi = int(hi/2)
		d = d*2

		xoff = (c_wid-(d*wi))/2
		yoff = (c_hi-(d*hi))/2
		
def set_res():
	global c_wid,c_hi,w,h,res,chosen_res,box,xoff,yoff,plotpoint,redraw

	w = res[chosen_res][0]
	h = res[chosen_res][1]
	box = res[chosen_res][2]
	xoff = res[chosen_res][3]
	yoff = res[chosen_res][4]
	
	if chosen_res:
		plotpoint = lowresplot
		redraw = low_res_redraw
	else:
		plotpoint = hiresplot
		redraw = handle_redraw
def choose_res():
	global chosen_res
	try:
		chosen_res = int(appuifw.popup_menu(res_lables,u"Set Resolution"))
		set_res()
	except:
		pass

def rule_gen():

	global ruletext	

	if automaton.name == "sim_1d":
		valid_vars = "e","c","w"

	if automaton.name == "nn_1d":
		valid_vars = "ee","e","c","w","ww"

	if automaton.name == "sim_2d":
		valid_vars = "c","n","e","s","w"

	if automaton.name == "nn_sim_2d":
		valid_vars = "c","n","ne","e","se","s","sw","w","nw"

	var_count = len(valid_vars)
	rule_bits = pow(2,var_count)

	ruletext = appuifw.query(u"Enter a function:","text",unicode(ruletext)).lower()

	ruletext = re.sub("v","(n+s+w+e)",ruletext)
	ruletext = re.sub("m","(n+s+w+e+ne+nw+se+sw)",ruletext)

	if ruletext:

		rp_rule = parse(ruletext,valid_vars)

		if rp_rule[0].type == 5:
			appuifw.note(u"Invalid rule!","error")
		else:
			var_values = dict()

			rule = 0

			progress = ProgressBar(images[orientation])

			for bit in range(rule_bits):

				grind = (bit*100)/rule_bits
				if grind % 5 == 0:
					progress.set_value(grind)
					handle_redraw(progress.rect)
						
				for sym in range(var_count):
					if pow(2,sym) & bit:
						this_value = 1
					else:
						this_value = 0
					var_values[valid_vars[sym]] = this_value

				if rp_eval(rp_rule,var_values):
					rule |= pow(2,bit)

			progress.close()
			del progress

			if automaton.name == "nn_sim_2d":
				rule = int2code(rule)

			automaton.set_rule(rule)	

			appuifw.note(unicode(str(rule)),"info")
		
def help():
	how = appuifw.popup_menu([u"About SCAMP",u"About this automaton",u"R.T.F.M."],u"Help!")

	if how == 0:
		appuifw.note(u"SCAMP, Copyright Dr Giles R. Greenway 2011, distributed under GNU GPL V3.","info")	

	if how == 1:
		openurl(docurl+automaton.name+'.html')

	if how == 2:
		openurl(docurl+'scamp.html')

def main_menu():
	choices = [(u"Select Automaton", automata),(u"Parameters", param), (u"Draw", draw)]

	if automaton.seed_type:
		choices += [(u"Seed",seed), (u"Seed Parameters",seedparam)]

	if automaton.recordable:
		choices += [(u"Draw and Record", draw_rec)]

	if automaton.prog:
		choices += [(u"Create Rule", rule_gen)]

	choices += [(u"palette",set_pal), (u"Resolution",choose_res),(u"Set Background",set_bg_col),(u"Save", save), (u"Path", img_dir),(u"Help",help)]

	appuifw.app.menu = choices

types = []
ca_lables = []
ca_names = []

seed_lables_1d = []
seed_lables_2d = []
seed_names_1d = []
seed_names_2d = []

ca_proto = [(u"Simple 1-D",[(u"rule",30,0,255),(u"history",0,-8,16)],sim_1d,"sim_1d",1,1,1),
(u"Next-Nearest 1-D",[(u"rule",1199303982,0,4294967296),(u"history",0,-8,8)],nn_1d,"nn_1d",1,1,1),
(u"Totalistic 1-D",[(u"order",3,2,255),(u"rule",65,0,9999)],total_1d,"total_1d",1,1,0),
(u"Pascal's Triangle",[(u"modulo",2,2,999),(u"colours",0,0,1)],pascal,"pascal",0,0,0),
(u"Skewed Pascal's Triangle",[(u"modulo",2,2,999),(u"colours",0,0,1),(u"history",0,-8,8)],skew_pascal,"skew_pascal",1,1,0),
(u"Cyclic 1-D",[(u"order",4,2,255),(u"neighbours",0,0,1)],cyclic_1d,"cyclic_1d",1,0,0),
(u"Simple 2-D",[(u"rule",2523490710,0,4294967296),(u"speed",1,0,1),(u"history",0,-8,8)],sim_2d,"sim_2d",1,2,1),
(u"Next-Nearest 2-D",[(u"code",u"1084qpqmqppFqpqmqmCmqpqmqg",0,0),(u"speed",1,0,1),(u"history",0,-8,8)],nn_sim_2d,"nn_sim_2d",1,2,1),
(u"Totalistic 2-D",[(u"d_rule",38,0,512),(u"l_rule",38,0,512),(u"neighbours",0,0,1),
(u"speed",1,0,1),(u"history",0,-8,8)],total_2d,"total_2d",1,2,0),
(u"Conway's Life",[(u"occupancy",300,0,999),(u"speed",1,0,1),(u"history",0,-8,16)],conlife,"conlife",1,0,0),
(u"Brian's Brain",[(u"occupancy",300,0,999),(u"neighbours",1,0,1),(u"speed",1,0,1)],bbrain,"bbrain",1,2,0),
(u"Cyclic Demons",[(u"order",5,2,255),(u"neighbours",0,0,1)],cyc_demon,"cyc_demon",1,0,0),
(u"Langton's Ant",[(u"order",2,2,255),(u"number",2,0,99999),(u"neighbours",0,0,1),(u"dt",1,1,999)],langton,"langton",1,2,0),(u"Turmites",[(u"rule",44355,0,65536),(u"neighbours",0,0,1),(u"history",0,-8,16),(u"dt",1,1,999)],turmites,"turmites",1,2,0),(u"B-Z reaction",[(u"order",16,2,255),
(u"k1",2,0,100),(u"k2",3,0,100),(u"g",20,0,100),(u"neighbours",1,0,1)],bzr,"bzr",1,0,0)]

for proto in ca_proto:
	ca_lables.append(proto[0])
	ca_names.append(proto[3])

one_d_seeds = [ca_seed(1,u"Single Defects","singledefect",[(u"number",1,0,64),(u"separation",0,0,128),(u"justify",0,0,2)]),ca_seed(1,u"Alternating Blocks","altblocks",[(u"on",2,1,64),(u"off",2,1,64),(u"number",0,0,64)]),ca_seed(1,u"Random","rand",[(u"probability",500,0,1000),(u"width",0,0,240)]),ca_seed(1,u"Bit String","bits",
[(u"left",0,0,255),(u"right",0,0,255),(u"justify",0,0,2)])]

two_d_seeds = [ca_seed(2,u"Single Defects","singledefect",[(u"xnumber",1,0,64),(u"ynumber",1,0,64),(u"xsep",0,0,128),(u"ysep",0,0,128)]),
ca_seed(2,u"Tabula Rasa","tabularasa",[(u"null",0,0,0)]),ca_seed(2,u"Random","rand",[(u"probability",500,0,1000)]),
ca_seed(2,u"Gradient","gradient",[(u"x0",1000,0,1000),(u"x1",0,0,1000),(u"y0",1000,0,1000),(u"y1",0,0,1000)]),
ca_seed(2,u"Bit Map","bitmap",[(u"bits",u"362",-1,-1),(u"cols",1,1,64),(u"xjust",0,0,2),(u"yjust",0,0,2)])]

pal_list = (u"Sprectrum",[red,yellow,green,blue,violet]),(u"RGB",[red,green,blue]),(u"RYGB",[red,yellow,green,blue]),(u"Greys",[white,black]),(u"Fire",[red,midcol(red,yellow),yellow,white]),(u"Water",[blue,midcol(blue,green),green,white]),(u"Desert",[midcol(black,red),midcol(red,yellow),yellow,white,midcol(green,blue),blue]),(u"Wood",[midcol(black,red),midcol(red,yellow),yellow]),(u"Lava",[black,white,yellow,red,midcol(red,black)]),(u"Matrix",[black,green,midcol(green,white)])
pal_lables = []
colours = pal_list[0][1]
bg_col = 0

for pal in pal_list:
	pal_lables.append(pal[0])

max_1d = 2
chosen_ca = 0
chosen_seed = 0
automaton = ca_type(ca_proto[chosen_ca])
drawing = 0
click = 0
rec = 0
rec_inter = 1
rec_path = ""

docurl = "file:///E:/Python/scampdocs/"
imgdirs = []
for drive in [u'E',u'F',u'C']:
	if os.path.exists(drive+":\\Images"):
		imgdirs.append(drive+":\\Images")
	if os.path.exists(drive+"\\Python\\scampdocs"):
		docurl = "file:///"+drive+":/Python/scampdocs/"
	if os.path.exists(drive+":\\scampdocs"):
		docurl = "file:///"+drive+":/scampdocs/"

if len(imgdirs)>1:
	img_path = imgdirs[1]
else:
	if len(imgdirs) > 0:
		img_path = imgdirs[0]
	else:
		imgdirs.append(u'.\\')
		img_path = u'.\\'

fname = ""

appuifw.app.orientation = 'automatic'

appuifw.app.screen = 'large'

appuifw.directional_pad = False

images = [0, 0]

try:
	canvas = appuifw.Canvas(redraw_callback = handle_redraw,event_callback = handle_event,resize_callback = handle_resize)
except:
	canvas = appuifw.Canvas(redraw_callback = handle_redraw,event_callback = handle_event)

appuifw.app.body = canvas

c_wid, c_hi = canvas.size

plotpoint = hiresplot
redraw = handle_redraw


w = c_wid
h = c_hi
box = 1
dbox = 0

xoff = 0
yoff = 0

res_lables = []
find_res()

chosen_res = 0

set_res()

if c_wid < c_hi:
	orientation = 1
	images = [graphics.Image.new((c_hi, c_wid)),graphics.Image.new((c_wid, c_hi))]
else:
	orientation = 0
	images= [graphics.Image.new((c_wid, c_hi)),graphics.Image.new((c_hi, c_wid))]

images[0].clear(white)
images[1].clear(white)
appuifw.app.exit_key_handler = quit

main_menu()

handle_redraw((0,0,c_wid,c_hi))

app_lock = e32.Ao_lock()
app_lock.wait()


