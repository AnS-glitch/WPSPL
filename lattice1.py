import numpy as np
import random
class Block:
	def __init__(self,x,y,width,height):
		self.x = x
		self.y = y
		self.width = width
		self.height = height
			
	@property
	def area(self):
		return self.width * self.height		#Calculates the area of a block
	def subdivide(self):					#A mother block is divided into 4 daughter blocks
		x_split = np.random.uniform(0, self.width)
		y_split = np.random.uniform(0, self.height)
		#Width and height of left/right and top/bottom sections
		w1, w2 = x_split, self.width  - x_split
		h1, h2 = y_split, self.height - y_split
		#4 new blocks
		bl_tl = Block(self.x,      self.y,      w1, h1)
		bl_tr = Block(self.x + w1, self.y,      w2, h1)
		bl_bl = Block(self.x,      self.y + h1, w1, h2)
		bl_br = Block(self.x + w1, self.y + h1, w2, h2)
		return [bl_tl, bl_tr, bl_bl, bl_br]

def choose_block(blocks):				#choosing randomly which block to select depending on their areas
		weights = [block.area for block in blocks]
		return random.choices(blocks, weights, k=1)[0]	#k=1 means it returns 1 object and [0] unwraps the list and gives the block object directly



#def remove_block(children,q):				

blocks = [Block(0,0,1,1),] 	#Area = 1
#selected = choose_block(blocks)
#print(selected.area)
for step in range(20000):
    # 1. Pick a block based on weighted area
    chosen = choose_block(blocks)
    #print(f'Chosen Block area: {chosen.area}')
    
    # 2. Remove the old chosen block from our list
    blocks.remove(chosen)
    
    # 3. Subdivide it into 4 new blocks and add them to the list
    new_subblocks = chosen.subdivide()
    blocks.extend(new_subblocks)

print(f"Total blocks now: {len(blocks)}")
total_area = 0.0
for i, b in enumerate(blocks):
    total_area = total_area + b.area
    #print(f"Block {i}: x={b.x}, y={b.y}, w={b.width}, h={b.height}, area={b.area}")
print(f'\n Total Area: {total_area}')


